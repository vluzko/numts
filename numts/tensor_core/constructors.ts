import {tensor, Shape, errors, ArrayOptions} from '../tensor';
import {indexing} from './indexing';
import {utils} from '../utils';


/**
 * Create a tensor from JSON.
 * @param json - The JSON representation of the array.
 */
export function from_json(json: any): tensor {
    return from_nested_array(json.data, json.dtype);
}

/**
 * Create a tensor from a nested array of values.
 * @param {any[]} array - An array of arrays (nested to arbitrary depth). Each level must have the same dimension.
 * The final level must contain valid data for a tensor.
 * @param {string} dtype  - The type to use for the underlying array.
 *
 * @return {tensor}
 */
export function from_nested_array(arr: any[], dtype?: string): tensor {
    if (arr.length === 0) {
        return array([]);
    }

    const dimensions = utils._nested_array_shape(arr);
    let slice_iter = indexing.iorder_index_iterator(dimensions);

    const size = indexing.compute_size(dimensions);
    const array_type = utils.dtype_map(dtype);
    const data = new array_type(size);

    let ndarray = array(data, dimensions, { dtype: dtype, disable_checks: true });

    for (let indices of slice_iter) {
        const real_index = ndarray._compute_real_index(indices);
        ndarray.data[real_index] = utils._nested_array_value_from_index(arr, indices);
    }

    return ndarray;
}

/**
 * Create an n-dimensional array from an iterable.
 * @param iterable
 * @param shape
 * @param {string} dtype
 * @return {tensor}
 */
export function from_iterable(iterable: Iterable<number>, shape: Shape, dtype?: string) {
    const final_shape = indexing.compute_shape(shape);

    const size = indexing.compute_size(final_shape);
    const array_type = utils.dtype_map(dtype);
    const index_iterator = indexing.iorder_index_iterator(final_shape);
    const val_gen = iterable[Symbol.iterator]();
    let data = new array_type(size);
    const stride = indexing.stride_from_shape(final_shape);
    const initial_offset = 0;
    let i = 0;
    for (let index of index_iterator) {
        const real_index = indexing.index_in_data(index, stride, initial_offset);
        let val = val_gen.next();
        data[real_index] = val.value;
    }

    if (data.length !== size) {
        throw new errors.MismatchedShapeSize(`Iterable passed has size ${data.length}. Size expected from shape was: ${size}`);
    }

    return array(data, final_shape, { disable_checks: true, dtype: dtype });
}

/**
 * Produces an array of the desired shape filled with a single value.
 * @param {number} value                - The value to fill in.
 * @param shape - A numerical array or a number. If this is a number a one-dimensional array of that length is produced.
 * @param {string} dtype                - The data type to use for the array. float64 by default.
 * @return {tensor}
 */
export function filled(value: number, shape, dtype?: string): tensor {
    const final_shape = indexing.compute_shape(shape);

    const size = indexing.compute_size(final_shape);
    const array_type = utils.dtype_map(dtype);
    const data = new array_type(size).fill(value);

    return array(data, final_shape, { disable_checks: true, dtype: dtype });
}

/**
 * Return an array of the specified size filled with zeroes.
 * Equivalent to `filled`, but slightly faster.
 * @param {number} shape
 * @param {string} dtype
 * @return {tensor}
 */
export function zeros(shape, dtype?: string): tensor {
    const final_shape = indexing.compute_shape(shape);
    const size = indexing.compute_size(final_shape);
    const array_type = utils.dtype_map(dtype);
    const data = new array_type(size);

    return array(data, final_shape, { disable_checks: true, dtype: dtype });
}

/**
 * Return an array of the specified size filled with ones.
 * @param {number} shape
 * @param {string} dtype
 * @return {tensor}
 */
export function ones(shape: number[] | Uint32Array, dtype?: string): tensor {
    return filled(1, shape, dtype);
}

/**
 * Create an identity matrix of a given size.
 * @param m - The size of the identity matrix.
 * @param dtype - The dtype for the identity matrix.
 */
export function eye(m: number, dtype?: string): tensor {
    let array = zeros([m, m], dtype);
    for (let i = 0; i < m; i++) {
        array.s(1, i, i);
    }
    return array;
}

/**
 * Create a tensor containing the specified data
 * @param data
 * @param shape
 * @param options
 * @return {tensor}
 */
export function array(data, shape?, options?: ArrayOptions): tensor {
    let final_shape;
    let size;
    let dtype;

    if (shape === undefined) {
        shape = new Uint32Array([data.length]);
    }

    if (options && options.dtype) {
        dtype = options.dtype
    }

    if (options && options.disable_checks === true) {
        final_shape = shape;
        size = indexing.compute_size(shape);
    } else {
        if (!utils.is_numeric_array(data)) {
            throw new errors.BadData();
        }

        if (shape === undefined || shape === null) {
            final_shape = new Uint32Array([data.length]);
        } else {
            final_shape = indexing.compute_shape(shape);
        }

        // Compute length
        size = indexing.compute_size(final_shape);

        if (size !== data.length) {
            throw new errors.MismatchedShapeSize()
        }
    }

    const stride = indexing.stride_from_shape(final_shape);
    const offset = new Uint32Array(final_shape.length);
    const dstride = stride.slice();

    return new tensor(data, final_shape, offset, stride, dstride, size, dtype);
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
  
    return from_iterable(iter, shape, "int32");
}