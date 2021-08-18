"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.compare_to_python = exports.call_python = void 0;
const child_process_1 = require("child_process");
const FILE = `${__dirname}/numpy_tests.py`;
/**
 * Call a python function and extract the result from stdout.
 * Is it insane to do message passing this way? Yes. But I'm lazy.
 */
function call_python(function_name, args, kwargs) {
    const arg_string = args === undefined ? '' : args.join(' ');
    let kwarg_array = [];
    if (kwargs !== undefined) {
        kwargs.forEach((v, k) => {
            const joined = `${k}=${v}`;
            kwarg_array.push(joined);
        });
    }
    const kwarg_string = kwarg_array.join(' ');
    const command = `python ${FILE} ${function_name} ${arg_string} ${kwarg_string}`;
    const res = child_process_1.execSync(command);
    return res;
}
exports.call_python = call_python;
/**
 * Compare a numts call to a python call.
 */
function compare_to_python(numts_func, python_func, args) {
    const kwargs = new Map();
    const numts_result = numts_func(...args);
    const py_result = call_python(python_func, args, kwargs);
    throw new Error('Not implemented.');
}
exports.compare_to_python = compare_to_python;
//# sourceMappingURL=call_python.js.map