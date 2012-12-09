# Python -> C99, OpenCL, JS converter
# created by Massimo Di Pierro
# license: 2-clause BSD
# requires meta (always) and numpy+pyopencl (if Device used)
import logging
import os
import sys
import imp
import ast
import uuid
import threading
import tempfile
import shutil
from meta.decompiler import decompile_func

try:
    import numpy
    import pyopencl as pcl
except ImportError:
    pcl = False

__all__ = ['Compiler', 'C99Handler', 'JavaScriptHandler', 'Device']

locker = threading.Lock()

# handle differences between Python 2na and Python 3

python_version = sys.version_info[0]
if python_version == 3:
    def get_arg(item):
        return item.arg

    def get_code(func):
        return func.__code__

    def map_node(node):
        return {'TryExcept': 'Try'}.get(node, node)
else:
    def get_arg(item):
        return item.id

    def get_code(func):
        return func.func_code

    def map_node(node):
        return node

# define useful constants

C0 = """
#include "Python.h"
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif
"""
C2 = '  {"%(name)s", %(module_name)s_%(name)s, METH_VARARGS},\n'
C3 = """
static PyObject * %(module_name)s_%(name)s(PyObject *self, PyObject *args) {
  %(crettype)s retval;
%(vars)s
  if(!PyArg_ParseTuple(args, "%(types)s"%(vars2)s)) return NULL;\n
  retval = %(name)s(%(args)s);
  return Py_BuildValue("%(rettype)s", retval);
} """
C4 = """
static PyMethodDef %(module_name)s_Methods[] = {
%(funcs)s  {NULL, NULL}
};
DL_EXPORT(void) init%(module_name)s(void) {
  Py_InitModule("%(module_name)s", %(module_name)s_Methods);
}
"""

MAP_TYPES = {
    'unsigned': 'i',
    'unsigned int': 'i',
    'int': 'i',
    'long': 'l',
    'float': 'f',
    'double': 'd',
    'char': 'c',
    'short': 'h',
    'char*': 's',
    'PyObject*': 'O'
    }

# logic to dynamically create a C-module from C-code

def distutil_compile_and_import(module_name, code, build_dir=None):
    from distutils.core import setup, Extension
    from distutils.util import get_platform
    if not build_dir:
        build_dir = tempfile.mkdtemp()

    # write code to file
    file_name = os.path.join(build_dir, module_name + '.c')
    open(file_name, 'w').write(code)
    home_dir = os.getcwd()

    # turn file into module
    os.chdir(build_dir)
    extension = Extension(module_name, [module_name + '.c'])
    try:
        locker.acquire()
        setup(name=module_name,
              version='0.1',
              ext_modules=[extension],
              script_args=["-q", "build"],
              script_name="C.py",
              package_dir=build_dir)
    finally:
        os.unlink(file_name)
        os.chdir(home_dir)
        locker.release()

    # import and return the module
    lib = "lib.%s-%s" % (get_platform(), sys.version[0:3])
    sys.path.append(os.path.join(build_dir, 'build', lib))
    try:
        fp, pathname, description = imp.find_module(module_name)
        module = imp.load_module(module_name, fp, pathname, description)
    finally:
        shutil.rmtree(build_dir)
        if fp:
            fp.close()
    return module

# the C99 handler converts AST to C99 code

class C99Handler(object):

    sep = ' ' * 4
    special_functions = {
        'REFD': lambda args: '(*(%s))' % ', '.join(args),
        'ADDR': lambda args: '(&(%s))' % ', '.join(args),
        'CAST': lambda args: '(%s)(%s)' % (
            C99Handler.make_type(args[0]), args[1])
    }
    substitutions = {'NULL': 'NULL', 'True': '1', 'False': '0'}

    def make_types(self, types):
        for key, value in types.items():
            types[key] = self.make_type(value)
        return types

    @staticmethod
    def make_type(name):
        newname = stars = ''
        if name.startswith('global:'):
            newname, name = newname + '__global ', name[7:]
        elif name.startswith('local:'):
            newname, name = newname + '__local ', name[6:]
        if name.startswith('const:'):
            newname, name = newname + 'const ', name[6:]
        if name.startswith('usigned'):
            newname, name = newname + 'unsigned ', name[6:]
        while name.startswith('ptr_'):
            stars, name = stars + '*', name[4:]
        return newname + name + stars

    def list_expressions(self, items, pad):
        t_items = [self.t(item, pad) for item in items]
        return pad + ('\n' + pad).join(t for t in t_items if t.strip())

    def is_FunctionDef(self, item, pad, types):
        args = ', '.join('%s %s' % (
            self.make_type(types[get_arg(a)]), get_arg(a))
            for a in item.args.args)
        return 'void %s(%s) {\n/*@VARS*/\n%s\n}' % (
            item.name, args, self.t(item.body, pad + self.sep))

    def is_Name(self, item, pad):
        id = item.id
        if id in self.substitutions:
            return self.substitutions[id]
        elif not id in self.constants:
            return id
        else:
            return self.represent(self.constants[id])

    def is_TryExcept(self, item, pad):
        raise NotImplementedError

    def is_Break(self, item, pad):
        return 'break;'

    def is_Continue(self, item, pad):
        return 'continue;'

    def is_While(self, item, pad):
        if item.orelse:
            raise NotImplementedError
        return 'while (%(c)s) {\n%(d)s\n%(p)s}' % dict(
            c=self.t(item.test), d=self.t(item.body, pad + self.sep), p=pad)

    def is_For(self, item, pad):
        if not item.iter.func.id == 'range':
            raise NotImplementedError
        if item.orelse:
            raise NotImplementedError
        args = item.iter.args
        if len(args) == 1:
            start, stop, incr = 0, self.t(args[0], pad), 1
        elif len(args) == 2:
            start, stop, incr = self.t(args[0], pad), self.t(args[1], pad), 1
        elif len(args) == 3:
            start, stop, incr = self.t(
                args[0], pad), self.t(args[1], pad), self.t(args[2], pad)
        else:
            raise NotImplementedError
        if isinstance(item.target, ast.Name) and \
                not item.target.id in self.symbols:
            self.symbols[item.target.id] = 'long'
        return 'for (%(n)s=%(a)s; %(n)s<%(b)s; %(n)s+=%(c)s) {\n%(d)s\n%(p)s}'\
            % dict(n=self.t(item.target), a=start, b=stop, c=incr,
                   d=self.t(item.body, pad + self.sep), p=pad)

    def is_If(self, item, pad):
        code = 'if (%(c)s) {\n%(b)s\n%(p)s}' % dict(
            c=self.t(item.test), b=self.t(item.body, pad + self.sep), p=pad)
        if item.orelse:
            code += ' else {\n%(e)s\n%(p)s}' % dict(
                e=self.t(item.orelse, pad + self.sep), p=pad)
        return code

    def is_Compare(self, item, pad):
        if len(item.ops) != 1 or len(item.comparators) != 1:
            raise NotImplementedError
        return '(%s %s %s)' % (self.t(item.left),
                               self.t(item.ops[0]),
                               self.t(item.comparators[0]))

    def is_BoolOp(self, item, pad):
        if len(item.values) != 2:
            raise NotImplementedError
        return '(%s %s %s)' % (self.t(item.values[0]),
                               self.t(item.op),
                               self.t(item.values[1]))

    def is_Not(self, item, pad):
        return '!'

    def is_Or(self, item, pad):
        return '||'

    def is_And(self, item, pad):
        return '&&'

    def is_Eq(self, item, pad):
        return '=='

    def is_Gt(self, item, pad):
        return '>'

    def is_Lt(self, item, pad):
        return '<'

    def is_GtE(self, item, pad):
        return '>='

    def is_LtE(self, item, pad):
        return '<='

    def is_Is(self, item, pad):
        return '=='

    def is_IsNot(self, item, pad):
        return '!='

    def is_NotEq(self, item, pad):
        return '!='

    def is_Assign(self, item, pad):
        if len(item.targets) != 1:
            raise NotImplementedError
        left, right = item.targets[0], item.value
        if isinstance(left, ast.Name) and not left.id in self.symbols:
            if isinstance(right, ast.Call) and \
                    right.func.id.startswith('new_'):
                self.symbols[left.id] = right.func.id[4:]
                right = right.args[0] if right.args else None
            else:
                # guess type
                if isinstance(right, ast.Num) and isinstance(right.n, float):
                    self.symbols[left.id] = 'float'
                elif isinstance(right, ast.Num) and isinstance(right.n, int):
                    self.symbols[left.id] = 'int'
                elif isinstance(right, ast.Str):
                    self.symbols[left.id] = 'ptr_char'
                else:
                    raise RuntimeError('unkown C-type %s' % left.id)
        return '%s = %s;' % (
            self.t(item.targets[0]), self.t(right)) if right else ''

    def is_Call(self, item, pad):
        func = self.t(item.func)
        args = [self.t(a) for a in item.args]
        if func in self.special_functions:
            return self.special_functions[func](args)
        return '%s(%s)' % (func, ','.join(args))

    def is_List(self, item, pad):
        return '{%s}' % ', '.join(self.t(k, pad) for k in item.elts)

    def is_Tuple(self, item, pad):
        return self.is_List(item, pad)

    def is_Num(self, item, pad):
        return self.represent(item.n)

    def is_Str(self, item, pad):
        return self.represent(item.s)

    def is_UnaryOp(self, item, pad):
        return '(%s %s)' % (self.t(item.op), self.t(item.operand))

    def is_BinOp(self, item, pad):
        return '(%s %s %s)' % (
            self.t(item.left), self.t(item.op), self.t(item.right))

    def is_USub(self, item, pad):
        return '-'

    def is_Add(self, item, pad):
        return '+'

    def is_Sub(self, item, pad):
        return '-'

    def is_Mult(self, item, pad):
        return '*'

    def is_Div(self, item, pad):
        return '/'

    def is_Subscript(self, item, pad):
        return '%s[%s]' % (self.t(item.value), self.t(item.slice.value))

    def is_Attribute(self, item, pad):
        return '%s.%s' % (self.t(item.value), item.attr)

    def is_Return(self, item, pad):
        self.rettype = self.symbols.get(item.value.id, None)
        return 'return %s;' % self.t(item.value) if self.rettype else ''

    def is_Expr(self, item, pad):
        return self.t(item.value) + ';'

    def represent(self, item):
        if item is None:
            return self.substitutions['NULL']
        elif isinstance(item, str):
            return '"%s"' % item.replace('"', '\\"')
        else:
            return str(item)

    def __init__(self):
        self.constants = {}
        self.actions = {list: self.list_expressions}
        for key in dir(self):
            if key.startswith('is_'):
                node = map_node(key[3:])
                self.actions[getattr(ast, node)] = getattr(self, key)

    def convert(self, item, types, prefix=''):
        self.rettype = None  # the return type of the function None is void
        self.symbols = {}
        code = self.is_FunctionDef(item, '', types)
        vars = ''.join('%s%s %s;\n' % (self.sep, v, k)
                       for k, v in self.symbols.items())
        code = code.replace('/*@VARS*/', vars)
        if self.rettype is not None:
            code = self.rettype + code[4:]
        return prefix + code.replace('\n\n', '\n')

    def t(self, item, pad=''):
        return self.actions[type(item)](item, pad)

# the JavaScriptHandler converts AST to JavaScript code

class JavaScriptHandler(C99Handler):

    special_functions = {'new': lambda args: 'new %s' % ', '.join(args)}
    substitutions = {'NULL': 'null', 'True': 'true', 'False': 'false'}

    @staticmethod
    def make_type(self, name):
        return name

    def is_List(self, item, pad):
        return '[%s]' % ', '.join(self.t(k, pad) for k in item.elts)

    def is_Dict(self, item, pad):
        n, ks, vs = len(item.keys), item.keys, item.values
        return '{%s}' % ', '.join(
            self.t(ks[k], pad) + ':' + self.t(vs[k], pad) for k in range(n))

    def is_TryExcept(self, item, pad):
        code = 'try {\n%(b)s\n%(p)s}' % dict(
            b=self.t(item.body, pad + self.sep), p=pad)
        if len(item.handlers) != 1:
            raise NotImplementedError
        handler = item.handlers[0]
        code += ' catch(%(n)s) {\n%(e)s\n%(p)s}' % dict(
            n=self.t(handler.type, pad),
            e=self.t(handler.body, pad + self.sep), p=pad)
        return code

    def is_FunctionDef(self, item, pad):
        args = ', '.join(a.id for a in item.args.args)
        return 'var %s = function(%s) {\n%s\n%s}' % (
            item.name, args, self.t(item.body, pad + self.sep), pad)

    def is_Assign(self, item, pad):
        if len(item.targets) != 1:
            raise NotImplementedError
        left, right = item.targets[0], item.value
        if isinstance(left, ast.Name) and not left.id in self.symbols:
            if isinstance(right, ast.Call) and \
                    right.func.id.startswith('new_'):
                jstype = right.func.id[4:] + ' '
                right = right.args[0] if right.args else None
            else:
                jstype = 'var '
        return '%s%s = %s;' % (jstype, self.t(item.targets[0]),
                               self.t(right)) if right else ''

    def is_Lambda(self, item, pad):
        args = ', '.join(a.id for a in item.args.args)
        return 'function (%s) { %s }' % (
            args, self.t(item.body, pad + self.sep))

    def __init__(self):
        C99Handler.__init__(self)

    def convert(self, item, types, prefix=''):
        self.rettype = None  # the return type of the function None is void
        self.symbols = {}
        code = self.is_FunctionDef(item, '')
        return prefix + code.replace('\n\n', '\n')

    def t(self, item, pad=''):
        return self.actions[type(item)](item, pad)

# the class Compiler defines the decorator

class Compiler(object):

    def __init__(self, handler=None):
        self.functions = {}
        self.handler = handler or C99Handler()

    def define_kernel(self, name=None, **types):
        return self.define('kernel', name, **types)

    def define(self, _prefix='', name=None, **types):
        if _prefix == 'kernel':
            _prefix = '__kernel '
        types = self.handler.make_types(types)

        def wrap(func, types=types, name=name, prefix=_prefix):
            if name is None:
                name = func.__name__
            decompiled = decompile_func(func)
            self.functions[name] = dict(func=func,
                                        prefix=prefix,
                                        ast=decompiled,
                                        types=types,
                                        code=None)
            return func
        return wrap

    def convert(self, headers=False, constants=None, call=False):
        """
        Returns all decorated Python code converted to target language
        constants is a dict() of constants to be used in convesion
        headers = True creates the headers (in c99 case only)
        call = 'f' calls the function f (for JS only)
        """
        if constants:
            self.handler.constants.update(constants)
        defs, funcs = [], []
        for name, info in self.functions.items():
            code = self.handler.convert(info['ast'],
                                        info['types'],
                                        info['prefix'])
            info['code'] = code
            if headers:
                defs.append(code.split(' {', 1)[0] + ';')
            funcs.append(code)
        code = '\n\n'.join(defs + funcs)
        if call:
            code = code + '\n\n%s();' % call
        return code

    def compile(self, constants=None, includes=None):
        """
        Compiles all decotared code and returns an
        returns modules conatining compiled versions of those functions
        Attention: this function performs a temporary change of directory
        """
        if not includes:
            includes = []
        if self.handler.__class__ != C99Handler:
            raise NotImplementedError("Required a C99Handler")
        filename = 'mdpcl'
        code = self.convert(headers=True, constants=constants)
        python_source = C0+'\n'.join(includes)+'\n'
        python_source += code
        module_name = 'c' + str(uuid.uuid4()).replace('-', '')
        funcs = ''
        for key, info in self.functions.items():
            func = info['func']
            name = func.__name__
            code = get_code(func)
            nargs = code.co_argcount
            args = ','.join(code.co_varnames[:nargs])
            parsing_code = ''
            vars, types, vars2 = '', '', ''
            for k, v in info['types'].items():
                t = MAP_TYPES[v]
                vars += '  %s %s;' % (v,k)
                types += MAP_TYPES[v]
                vars2 += ', &%s' % k
            funcs += C2 % dict(name=name, module_name=module_name)
            crettype = self.handler.rettype
            rettype = MAP_TYPES[crettype]
            python_source += C3 % dict(module_name=module_name,
                                       filename=filename,
                                       name=name, args=args,
                                       vars = vars,
                                       types = types,
                                       vars2 = vars2,
                                       crettype = crettype,
                                       rettype = rettype)
        python_source += C4 % dict(module_name=module_name,
                                   filename=filename,
                                   name=name, args=args,
                                   parsing_code=parsing_code,
                                   funcs=funcs)
        module = distutil_compile_and_import(module_name, python_source)
        return module

# only for open, this is the main interface to the device

if pcl:
    class Device(object):

        flags = pcl.mem_flags

        def __init__(self):
            self.ctx = pcl.create_some_context()
            self.queue = pcl.CommandQueue(self.ctx)
            self.compiler = Compiler()

        def buffer(self, source=None, size=0,
                   mode=pcl.mem_flags.READ_WRITE):
            if source is not None:
                mode = mode | pcl.mem_flags.COPY_HOST_PTR
            buffer = pcl.Buffer(self.ctx, mode, size=size, hostbuf=source)
            return buffer

        def retrieve(self, buffer, shape=None, dtype=numpy.float32):
            output = numpy.zeros(shape or buffer.size / 4, dtype=dtype)
            pcl.enqueue_copy(self.queue, output, buffer)
            return output

        def compile(self, kernel=None, constants=None, includes=None):
            if not kernel:
                kernel = self.compiler.convert(headers=True,
                                               constants=constants)
            kernel = '\n'.join(includes or []) + '\n' + kernel
            return pcl.Program(self.ctx, kernel).build()


# some tests/examples

def test_c99():
    c99 = Compiler()

    @c99.define(a='int', b='int')
    def f(a, b):
        for k in range(n):
            while True:
                break
            if k == 0 or k != 0 or k < 0 or k > 0 or not k == 0 or \
                    k >= 0 or k <= 0 or k is None or k is not None:
                continue
        c = new_int(a + b)
        printf("%i + %i = %i", a, b, c)

        d = new_ptr_int(CAST(ptr_int, ADDR(c)))
        c = REFD(d)
        return c
    print(c99.convert(headers=False, constants=dict(n=10)))


def test_C_compile():
    c99 = Compiler()

    @c99.define(n='int')
    def factorial(n):
        output = 1
        for k in range(1, n + 1):
            output = output * k
        return output
    compiled = c99.compile()
    print(compiled.factorial(10))
    assert compiled.factorial(10) == factorial(10)


def test_OpenCL():
    if not pcl:
        logging.error('must install "pyopencl"')
        sys.exit(1)
    device = Device()

    @device.compiler.define_kernel(
        w='global:ptr_float',
        u='global:const:ptr_float',
        q='global:const:ptr_float')
    def solve(w, u, q):
        x = new_int(get_global_id(0))
        y = new_int(get_global_id(1))
        site = new_int(x * n + y)
        if y != 0 and y != n - 1 and x != 0 and x != n - 1:
            up = new_int(site - n)
            down = new_int(site + n)
            left = new_int(site - 1)
            right = new_int(site + 1)
            w[site] = 1.0 / 4 * (u[up] + u[down] + u[left] + u[
                                 right] - q[site])
    print(device.compiler.convert(headers=True, constants=dict(n=300)))


def test_JS():
    js = Compiler(handler=JavaScriptHandler())

    @js.define()
    def f(a):
        a = new(array(1, 2, 3, 4))
        v = [1, 2, 'hello']
        w = {'a': 2, 'b': 4}

        def g():
            try:
                alert('hello')
            except e:
                alert(e)
        jQuery('button').click(lambda: g())
    print(js.convert(call='f'))

if __name__ == '__main__':
    test_c99()
    test_C_compile()
    test_OpenCL()
    test_JS()
