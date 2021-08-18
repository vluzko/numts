import {execSync} from 'child_process';
import {tensor} from '../numts/tensor';

const FILE = `${__dirname}/numpy_tests.py`;

/**
 * Call a python function and extract the result from stdout.
 * Is it insane to do message passing this way? Yes. But I'm lazy.
 */
export function call_python(function_name: string, args: Array<string>, kwargs: Map<string, string>): any {
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
  const res = execSync(command);
  return res;
}


/**
 * Compare a numts call to a python call.
 */
export function compare_to_python(numts_func: any, python_func: string, args: any) {
    const kwargs = new Map();
    const numts_result = numts_func(...args);
    const py_result = call_python(python_func, args, kwargs);
    throw new Error('Not implemented.');
}

