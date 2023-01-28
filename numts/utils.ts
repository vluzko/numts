
type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array| Int32Array | Uint32Array | Float32Array | Float64Array;
type Numeric = TypedArray | number[];
type Shape = number[] | Uint32Array;


export namespace utils {
  /**
   * TODO: Move to a static function in tensor
   * @param array1
   * @param array2
   * @return {number}
   */
  export function dot(array1, array2): number {
    return array1.reduce((a, b, i) => a + b * array2[i], 0);
  }

  /**
   * Compare two numeric arrays.
   * @param {Numeric} a - The first array to compare.
   * @param {Numeric} b - Second array to compare.
   * @return {boolean}
   */
  export function array_equal(a: Numeric, b: Numeric): boolean {
    if (a.length !== b.length) {
      return false;
    } else {
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
          return false;
        }
      }
      return true;
    }
  }

  /**
   * TODO: Test
   * Checks whether a value is a number and isn't null.
   * @param value - The value to check.
   * @return {boolean}
   */
  export function is_numeric(value: any): value is number {
    return !isNaN(value) && value !== null && !ArrayBuffer.isView(value);
  }

  /**
   * Check whether a value is an integer.
   * @param {any} value - The value to check.
   * @return {boolean}
   */
  export function is_int(value: any): value is number {
    return Number.isInteger(value);
  }

  /**
   * Checks whether a value is an array(like) of numbers.
   * @param array
   * @return {boolean}
   * @private
   */
  export function is_numeric_array(array: any): boolean {
    if (!Array.isArray(array) && !ArrayBuffer.isView(array)) {
      return false;
    } else if (ArrayBuffer.isView(array)) {
      return true;
    } else {
        return (<number[]>array).reduce((a, b) => is_numeric(b) && a, true);
    }
  }

  export function zip_iterable(...iters: Iterator<any>[]): Iterable<any[]> {
    let iterators = iters.map(e => e[Symbol.iterator]());

    let iter = {};
    iter[Symbol.iterator] = function* () {
      let all_done = false;
      while (!all_done) {
        let results = [];
        iterators.forEach(e => {
          let {value, done} = e.next();
          if (done) {
            all_done = true;
          }
          results.push(value);
        });

        if (!all_done) {
          yield results;
        }
      }
    };

    return <Iterable<number[]>> iter;
  }

  export function zip_longest(...iters: Iterable<any>[]): Iterable<any[]> {
    let iterators = iters.map(e => e[Symbol.iterator]());

    let iter = {
      [Symbol.iterator]: function*() {
        let individual_done = iters.map(e => false);
        let all_done = false;
        while (!all_done) {
          let results = [];
          iterators.forEach((e, i) => {
            let {value, done} = e.next();
            if (done) {
              individual_done[i] = true;
              iterators[i] = iters[i][Symbol.iterator]();
              value = iterators[i].next()["value"];
            }
            results.push(value);
          });

          all_done = individual_done.reduce((a, b) => a && b);
          if (!all_done) {
            yield results;
          }
        }
      }
    };

    return <Iterable<number[]>> iter;
  }

  // TODO: Test
  /**
   * Check if value is an ArrayBuffer
   * @param value
   * @return {boolean}
   */
  export function is_typed_array(value: any): value is TypedArray {
    return !!(value.buffer instanceof ArrayBuffer && value.BYTES_PER_ELEMENT);
  }

  /**
   * Subtract two typed arrays. Should only be called on typed arrays that are guaranteed to be the same size.
   * @param {TypedArray} a
   * @param {TypedArray} b
   * @return {TypedArray}
   * @private
   */
  export function _typed_array_sub(a: Numeric, b: Numeric) {
    // @ts-ignore
    return a.map((e, i) => e - b[i]);
  }

  /**
   * Convert a dtype string to the corresponding TypedArray constructor.
   * @param dtype
   * @return {any}
   * @private
   */
  export function dtype_map(dtype: string) {
    let array_type;
    switch (dtype) {
      case "int8":
        array_type = Int8Array;
        break;
      case "int16":
        array_type = Int16Array;
        break;
      case "int32":
        array_type = Int32Array;
        break;
      case "uint8":
        array_type = Uint8Array;
        break;
      case "uint8c":
        array_type = Uint8ClampedArray;
        break;
      case "uint16":
        array_type = Uint16Array;
        break;
      case "uint32":
        array_type = Uint32Array;
        break;
      case "float32":
        array_type = Float32Array;
        break;
      case "float64":
        array_type = Float64Array;
        break;
      default:
        array_type = Float64Array;
    }

    return array_type;
  }

  /**
   *
   * @param {string} a  - The first dtype.
   * @param {string} b  - The second dtype.
   * @return {string} - The smallest dtype that can contain a and b without losing data.
   * @private
   */
  export function _dtype_join(a: string, b: string): string {
    // type TypedArray = Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array| Int32Array | Uint32Array | Float32Array | Float64Array;
    const ordering = [["int8", "uint8", "uint8c"], ["int16", "uint16"], ["int32", "uint32", "float32"], ["float64"]];
    const a_index = ordering.reduce((acc, e, i) => e.indexOf(a) === -1 ? acc : i, -1);
    const b_index = ordering.reduce((acc, e, i) => e.indexOf(b) === -1 ? acc : i, -1);
    if (a === b) {
      return a;
    } else if (a_index === b_index) {
      return ordering[a_index + 1][0];
    } else if (a_index < b_index) {
      return b;
    } else {
      return a;
    }
  }

  /**
   * Get lower indices and steps for a slice over an entire shape.
   * @param shape - The shape of the array.
   */
  export function shape_to_lus(shape: Uint32Array): [Uint32Array, Uint32Array, Uint32Array] {
    const lower = new Uint32Array(shape.length);
    let steps = new Uint32Array(shape.length);
    steps.fill(1);
    return [lower, shape.slice(0), steps];
  }

  /**
   * Create a fixed-sized array of ones.
   * @param length - The length of the array.
   */
  export function fixed_ones(length: number): Uint32Array {
    let steps = new Uint32Array(length);
    steps.fill(1);
    return steps;
  }

  /**
   * Add one to a mixed-radix number represented by a Uint32Array.
   * Note that it is assumed that it is *possible* to add one to the given value. This function will *not* extend the array.
   * @param value - The current value of the number.
   * @param upper_bounds - The upper bounds of each digit. In other words the base for the numerical system used in that column
   * @param index - The index to increment.
   */
  export function increment_digit(value: Uint32Array, upper_bounds: Uint32Array, index: number): Uint32Array {
    let cur_index = index;
    let copy = value.slice(0);
    let done = false;
    while (!done) {
      if (copy[cur_index] === upper_bounds[cur_index]) {
        copy[cur_index] = 0;
        cur_index -= 1;
      } else {
        copy[cur_index] = copy[cur_index] + 1;
        done = true;
      }
    }
    return copy;
  }

  export function imap<T, S>(iter: Iterable<T>, f: (a: T) => S): Iterable<S> {
    const new_iter = {
      [Symbol.iterator]: function* () {
        for (let i of iter) {
          yield f(i);
        }
      }
    };
    return new_iter;
  }

  /**
 * Compute the dimensions of a nested array.
 * @param {any[]} nested_array  - Arrays nested arbitrarily deeply. Each array of the same depth must have the same length.
 *                                This is *not* checked.
 * @return {Uint32Array}        - The dimensions of each subarray.
 * @private
 */
export function _nested_array_shape(nested_array: any[]): Uint32Array {
  let dims: number[] = [];
  let current = nested_array;
  let at_bottom = false;
  while (!at_bottom) {
    dims.push(current.length);
    if (Array.isArray(current[0])) {
      current = current[0];
    } else {
      at_bottom = true;
    }
  }

  return new Uint32Array(dims);
}

/**
 *
 * @param {any[]} nested_array
 * @param {Uint32Array} indices
 * @return {any[]}
 * @private
 */
export function _nested_array_value_from_index(nested_array: any[], indices: Uint32Array) {
  let current_subarray = nested_array;
  for (let index of indices) {
    current_subarray = current_subarray[index];
  }
  return current_subarray;
}
}
