import {tensor} from "./tensor";
import {indexing} from "./indexing";
import {utils} from "./utils";
import {Shape} from "./types";

/**
 * Return indices
 * @param condition
 * @param a
 * @param b
 */
export function where(condition, a, b?) {
  throw new Error();
}

// /**
//  *  Return an array of booleans. Each entry is whether the corresponding entries in a and b are numerically close. The arrays will be broadcasted. 
//  * @param a - First array to compare.
//  * @param b - Second array to compare. 
//  * @param rel_tol - The maximum relative error.
//  * @param abs_tol - The maximum absolute error.
//  */
// export function isclose(a: tensor, b: tensor, rel_tol: number = 1e-5, abs_tol: number = 1e-8): tensor {
//     const compare = (x: number, y: number): number => {
//         return +(Math.abs(x - y) <= abs_tol + (rel_tol * Math.abs(y)));
//     }
//     return tensor._binary_broadcast(a, b, compare);
// }

// TODO: Allow non-tensor arrays
// TODO: Type upcasting.
// /**
//  * Compute the sum of two arrays.
//  * output[i] = a[i] + [i].
//  * @param a
//  * @param b
//  * @return {number | tensor}
//  */
// export function add(a, b) {
//   return tensor._add(a, b);
// }

// export function div(a, b) {
//   return tensor._div(a, b);
// }

// export function mult(a, b) {
//   return tensor._mult(a, b);
// }

// export function sub(a, b) {
//   return tensor._sub(a, b);
// }

/**
 * Wrapper around tensor.zeros
 * @param {number[] | Uint32Array} shape
 * @param {string} dtype
 * @return {tensor}
 */
export function zeros(shape: number[] | Uint32Array, dtype?: string): tensor {
  return tensor.zeros(shape, dtype);
}

/**
 * Return an array of the specified size filled with ones.
 * @param {number} shape
 * @param {string} dtype
 * @return {tensor}
 */
export function ones(shape: number[] | Uint32Array, dtype?: string): tensor {
  return tensor.filled(1, shape, dtype);
}

/**
 * Create a tensor containing a range of integers.
 * @param {number} start_or_stop  - If no other arguments are passed, the upper bound of the range (with lower bound zero). Otherwise this is the lower bound.
 * @param {number} stop           - The upper bound of the range.
 * @param {number} step           - The step size between elements in the range.
 * @param {Shape} shape           - The shape to return.
 * @return {tensor}             - A one-dimensional array containing the range.
 */
export function arange(start_or_stop: number, stop?: number, step?: number, shape?: Shape): tensor {
  if (step === undefined) {
    step = 1;
  }
  
  let start;
  if (stop === undefined) {
    stop = start_or_stop;
    start = 0;
  } else {
    start = start_or_stop;
  }
  
  let size = Math.abs(Math.floor((stop - start) / step));
  if (shape === undefined) {
    shape = new Uint32Array([size]);
  } else {
    const shape_size = indexing.compute_size(shape);
    if (shape_size !== size) {
      throw new Error(`Mismatch between size of range (${size}) and size of shape (${shape_size}`);
    }
  }

  let iter = {
    [Symbol.iterator]: function*() {
      let i = start;
      while (i < real_stop) {
        yield i;
        i += step;
      }
    }
  };
  
  let real_stop = stop < start ? -stop : stop;
  
  return tensor.from_iterable(iter, shape, "int32");
}

/**
 * Create a tensor from a nested array of values.
 * @param {any[]} array - An array of arrays (nested to arbitrary depth). Each level must have the same dimension.
 * The final level must contain valid data for a tensor.
 * @param {string} dtype  - The type to use for the underlying array.
 *
 * @return {tensor}
 */
export function from_nested_array(array: any[], dtype?: string): tensor {
  return tensor.from_nested_array(array, dtype); 
}

export {tensor as tensor};
