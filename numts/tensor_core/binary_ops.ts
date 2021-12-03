import {tensor, Broadcastable} from '../tensor';
import * as constructors from './constructors';
import {indexing} from '../indexing';
import {utils} from '../utils';

/**
 * Convert a broadcastable value to a tensor.
 * @param {Broadcastable} value - The value to convert. Numbers will be converted to 1x1 tensors, TypedArrays will be 1xn, and tensors will be left alone.
 * @return {tensor}           - The resulting tensor.
 * @
 */
export function _upcast_to_tensor(value: Broadcastable): tensor {
    let a_array;
    if (utils.is_numeric(value)) {
        a_array = constructors.array(new Float64Array([value]), new Uint32Array([1]), { disable_checks: true });
    } else if (utils.is_typed_array(value)) {
        a_array = constructors.array(value, new Uint32Array([value.length]), { disable_checks: true });
    } else {
        a_array = value;
    }
    return a_array;
}

/**
 * Broadcast two values together.
 * Works like numpy broadcasting.
 * @param {Broadcastable} a - The first broadcastable value.
 * @param {Broadcastable} b - The second broadcastable value.
 * @return {[IterableIterator<number[]>, Uint32Array, string]}  - An iterator over that returns a tuple (a_i, b_i) of broadcasted values, the new shape, and the new dtype.
 * @
 */
export function _broadcast_by_index(a: Broadcastable, b: Broadcastable): [IterableIterator<[number, number, Uint32Array]>, Uint32Array, string] {

    let a_array = _upcast_to_tensor(a);
    let b_array = _upcast_to_tensor(b);

    const new_dimensions = indexing.calculate_broadcast_dimensions(a_array.shape, b_array.shape);
    const new_dtype = utils._dtype_join(a_array.dtype, b_array.dtype);
    let index_iter = indexing.iorder_index_iterator(new_dimensions);

    const iterator = utils.zip_longest(a_array._iorder_data_iterator(), b_array._iorder_data_iterator(), index_iter);

    let iter = {};
    iter[Symbol.iterator] = function* () {
        for (let [a_index, b_index, index] of iterator) {
            const a_val = a_array.data[a_index];
            const b_val = b_array.data[b_index];
            yield [a_val, b_val, index];
        }
    };

    return [<IterableIterator<[number, number, Uint32Array]>>iter, new_dimensions, new_dtype];
}

/**
 * Apply a binary function to two broadcastables.
 * @param {Broadcastable} a - The first argument to f.
 * @param {Broadcastable} b - The second argument to f.
 * @param {(a: number, b: number) => number} f  - The function to apply.
 * @param {string} dtype  - Optional forced data type.
 * @return {tensor}  - The result of applying f to a and b.
 * @
 */
export function _binary_broadcast(a: Broadcastable, b: Broadcastable, f: (a: number, b: number) => number, dtype?: string): tensor {
    let [iter, shape, new_dtype] = _broadcast_by_index(a, b);

    if (dtype === undefined) {
        dtype = new_dtype
    }

    let new_array = constructors.filled(0, shape, dtype);

    for (let [a_val, b_val, index] of iter) {
        const new_val = f(a_val, b_val);
        new_array.s(new_val, ...index);
    }

    return new_array
}

/**
 * 
 * @param {Broadcastable} a -
 * @param {Broadcastable} b -
 * @returns {tensor}      -
 */
export function broadcast_matmul(a: Broadcastable, b: Broadcastable): tensor {
    let a_array = _upcast_to_tensor(a);
    let b_array = _upcast_to_tensor(b);

    const a_shape: Uint32Array = a_array.shape;
    const b_shape: Uint32Array = b_array.shape;

    // Check they can actually be multiplied.
    if (a_shape[a_shape.length - 1] !== b_shape[b_shape.length - 2]) {
        throw new Error(`Shapes ${a_shape} and ${b_shape} are not aligned for matrix multiplication.`);
    }

    const broadcast = indexing.calculate_broadcast_dimensions(a_array.shape.slice(0, -2), b_array.shape.slice(0, -2));
    const new_dimensions = new Uint32Array([...broadcast,
    a_shape[a_shape.length - 2],
    b_shape[b_shape.length - 1]
    ]);

    if (new_dimensions.length === 2) {
        return tensor.matmul_2d(a_array, b_array);
    } else {
        const new_dtype = utils._dtype_join(a_array.dtype, b_array.dtype);
        let array = constructors.zeros(new_dimensions, new_dtype);

        const index_iter = indexing.iorder_index_iterator(new_dimensions.slice(0, -2));
        const a_iter = indexing.iorder_index_iterator(a_shape.slice(0, -2));
        const b_iter = indexing.iorder_index_iterator(b_shape.slice(0, -2));
        const iter = utils.zip_longest(a_iter, b_iter, index_iter);
        for (let [a_index, b_index, index] of iter) {
            const slice = indexing.index_to_slice(index);

            const b1 = b_array.slice(...b_index);
            const a1 = a_array.slice(...a_index);
            const subarray = tensor.matmul_2d(a1, b1);

            array.s(subarray, ...slice);
        }
        return array;
    }
}

/**
 * Multiply two 2D matrices.
 * Computes a x b.
 * @param {tensor} a  - The first array. Must be m x n.
 * @param {tensor} b  - The second array. Must be n x p.
 * @returns {tensor}  - The matrix product.
 */
export function matmul_2d(a: tensor, b: tensor): tensor {
    const new_shape = new Uint32Array([a.shape[0], b.shape[1]]);

    let iter = {
        [Symbol.iterator]: function* () {
            for (let i = 0; i < new_shape[0]; i++) {
                for (let j = 0; j < new_shape[1]; j++) {
                    const a_vec = a.slice(i);
                    const b_vec = b.slice(null, j);
                    let x = tensor.dot(a_vec, b_vec);
                    yield x;
                }
            }
        }
    };

    return constructors.from_iterable(iter, new_shape);
}

// TODO: Generalize to an inner product.
// TODO: This is numerically unstable.
/**
 * Compute the dot product of two arrays.
 * @param {tensor} a
 * @param {tensor} b
 * @return {number}
 */
export function dot(a: tensor, b: tensor): number {
    let acc = 0;
    let a_iter = a._iorder_value_iterator();
    let b_iter = b._iorder_value_iterator();
    for (let [a_val, b_val] of utils.zip_iterable(a_iter[Symbol.iterator](), b_iter[Symbol.iterator]())) {
        acc += a_val * b_val;
    }
    return acc;
}

/**
 * Create an array containing the element-wise max of the inputs.
 * Inputs must be the same shape.
 * @param {tensor} a  - First array.
 * @param {tensor} b  - Second array.
 * @return {tensor}   - An array with the same shape as a and b. Its entries are the max of the corresponding entries of a and b.
 */
export function take_max(a: tensor, b: tensor) {
    return _binary_broadcast(a, b, (x, y) => Math.max(x, y));
}

/**
 * Create an array containing the element-wise min of the inputs.
 * Inputs must be the same shape.
 * @param {tensor} a  - First array.
 * @param {tensor} b  - Second array.
 * @return {tensor}   - An array with the same shape as a and b. Its entries are the min of the corresponding entries of a and b.
 */
export function take_min(a: tensor, b: tensor) {
    return _binary_broadcast(a, b, (x, y) => Math.min(x, y));
}

/**
 * Compute the sum of two arrays.
 * output[i] = a[i] + [i].
 * @param a
 * @param b
 * @return {number | tensor}
 */
export function _add(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => x + y);
}

/**
 * Subtract an array from another.
 * output[i] = a[i] - b[i].
 * @param {Broadcastable} a - The minuend.
 * @param {Broadcastable} b - The subtrahend.
 * @return {Broadcastable} - The element-wise difference.
 */
export function _sub(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => x - y);
}

/**
 * Compute the Hadamard product of two arrays, i.e. the element-wise product of the two arrays.
 * output[i] = a[i] * b[i].
 * @param {Broadcastable} a - First factor.
 * @param {Broadcastable} b - Second factor.
 * @return {Broadcastable} - The element-wise product of the two inputs.
 */
export function _mult(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => x * y);
}

/**
 * Compute the element-wise quotient of the two inputs.
 * output[i] = a[i] / b[i].
 * @param {Broadcastable} a - Dividend array.
 * @param {Broadcastable} b - Divisor array.
 * @return {Broadcastable}  - Quotient array.
 */
export function _div(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => x / y, 'float64');
}

/**
 * Compute the element-wise power of two inputs
 * @param {Broadcastable} a - Base array.
 * @param {Broadcastable} b - Exponent array.
 * @return {tensor}       - Result array.
 * @
 */
export function _power(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => Math.pow(x, y), 'float64');
}

/**
 * Compute the element-wise quotient of two arrays, rounding values up to the nearest integer.
 * @param {Broadcastable} a - Dividend array.
 * @param {Broadcastable} b - Divisor array.
 * @return {Broadcastable}  - Quotient array.
 */
export function _cdiv(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => Math.ceil(x / y));
}

/**
 * Compute the element-wise quotient of two arrays, rounding values down to the nearest integer.
 * @param {Broadcastable} a - Dividend array.
 * @param {Broadcastable} b - Divisor array.
 * @return {tensor}       - Quotient array.
 */
export function _fdiv(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => Math.floor(x / y));
}

/**
 * Compute element-wise modulus of two arrays.
 * @param {Broadcastable} a - First array.
 * @param {Broadcastable} b - Second array.
 * @return {tensor}       - Modulus array.
 */
export function _mod(a: Broadcastable, b: Broadcastable): tensor {
    return _binary_broadcast(a, b, (x, y) => x % y);
}

/**
 * Compute element-wise less than.
 * @param {tensor} a
 * @param {tensor} b
 */
export function _lt(a: tensor, b: tensor) {
    return _binary_broadcast(a, b, (x, y) => +(x < y), 'uint8');
}

/**
 * Compute element-wise greater than.
 * @param {tensor} a
 * @param {tensor} b
 */
export function _gt(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => +(x > y), 'uint8');
}

/**
 * Compute element-wise less than or equal to.
 * @param {Broadcastable} a
 * @param {Broadcastable} b
 */
export function _le(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => +(x <= y), 'uint8');
}

/**
 * Compute element-wise greater than or equal to.
 * @param {Broadcastable} a
 * @param {Broadcastable} b
 */
export function _ge(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => +(x >= y), 'uint8');
}

/**
 * Compute element-wise not equal to.
 * @param {Broadcastable} a
 * @param {Broadcastable} b
 */
export function _ne(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => +(x !== y), 'uint8');
}

/**
 * Compute element-wise equality.
 * @param {Broadcastable} a
 * @param {Broadcastable} b
 */
export function _eq(a: Broadcastable, b: Broadcastable) {
    return _binary_broadcast(a, b, (x, y) => +(x === y), 'uint8');
}

export function is_close(a: tensor, b: tensor, rel_tol: number = 1e-5, abs_tol: number = 1e-8): tensor {
    const compare = (x: number, y: number): number => {
        return +(Math.abs(x - y) <= abs_tol + (rel_tol * Math.abs(y)));
    }
    return _binary_broadcast(a, b, compare);
}