import {tensor, TypedArray} from '../tensor';
import * as constructors from './constructors';
import {indexing} from './indexing';
import {utils} from '../utils';

/**
 * Return a copy of the tensor cast to the specified type.
 */
export function _as_type(a: tensor, dtype: string): tensor {
    const array_type = utils.dtype_map(dtype);
    const new_data = new array_type(a.data.slice(0));
    return new tensor(new_data, a.shape.slice(0), a.offset.slice(0), a.stride.slice(0), a.dstride.slice(0), a.length, dtype, a.is_view, a.initial_offset);
}

/**
 * Clip all values in the array to be in the specified range.
 * @param lower - The lower bound of the range.
 * @param upper - The upper bound of the range.
 */
export function _clip(a: tensor, lower: number, upper: number): tensor {
    return a.map(e => {
        if (e < lower) {
            return lower;
        } else if (e > upper) {
            return upper;
        } else {
            return e;
        }
    });
}


/**
 * Create a copy of this with a different shape.
 * @param {Uint32Array} new_shape - The shape to make the new array.
 * @return {tensor}             - The reshaped array.
 */
export function _reshape(a: tensor, ...new_shape: Array<Uint32Array | number[] | number>): tensor {
    let shape: Uint32Array | number[];
    if (utils.is_numeric_array(new_shape[0])) {
        // @ts-ignore
        shape = new_shape[0];
    } else {
        // @ts-ignore
        shape = new_shape;
    }

    if (Array.isArray(shape)) {
        shape = new Uint32Array(shape);
    }

    const new_size = indexing.compute_size(shape);
    const size = indexing.compute_size(a.shape);
    if (size !== new_size) {
        throw new Error(`Array cannot be reshaped because sizes do not match. Size of underlying array: ${size}. Size of reshaped array: ${shape}`);
    }
    let value_iter = a._iorder_value_iterator();
    return constructors.from_iterable(value_iter, shape, a.dtype);
}

/**
 * Flatten an array. Elements will be in iteration order.
 * @returns - The flattened array
 */
export function _flatten(a: tensor): tensor {
    const shape = new Uint32Array([a.length]);
    return constructors.from_iterable(a._iorder_value_iterator(), shape, a.dtype);
}

/**
 * Returns the negation of this array.
 */
export function _neg(a: tensor): tensor {
    const new_data = a.data.map(x => -x);
    return constructors.array(new_data, a.shape, { disable_checks: true, dtype: a.dtype });
}

/**
 * Return the transpose of this array.
 */
export function _transpose(a: tensor): tensor {
    const new_shape = a.shape.slice(0).reverse();
    let new_array = constructors.zeros(new_shape, a.dtype);
    for (let index of a._iorder_index_iterator()) {
        const value = a.g(...index);
        const new_index = index.reverse();
        new_array.s(value, ...new_index);
    }
    return new_array;
}

/**
 * Extract the upper triangle of this tensor.
 */
export function _triu(a: tensor): tensor {
    const iter = utils.imap(a._iorder_index_iterator(), i => {

        if (i[i.length - 2] <= i[i.length - 1]) {
            return a.g(...i);
        } else {
            return 0;
        }
    });
    return constructors.from_iterable(iter, a.shape, a.dtype);
}

/**
 * Extract the lower triangle of this tensor.
 */
export function _tril(a: tensor): tensor {
    const iter = utils.imap(a._iorder_index_iterator(), i => {

        if (i[i.length - 2] >= i[i.length - 1]) {
            return a.g(...i);
        } else {
            return 0;
        }
    });
    return constructors.from_iterable(iter, a.shape, a.dtype);
}

export function _round() {

}
