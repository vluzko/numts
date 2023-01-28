const constructors = require('../../numts/tensor_core/constructors');
const utils = require('../../numts/utils').utils;

describe('array.', function () {
    test('Simple.', function() {
        const data = new Uint32Array([1, 2])
        const a = constructors.array(data);
        expect(a.data).toEqual(data);
        expect(a.shape).toEqual(new Uint32Array([2]));
    })
})

describe('zeros.', function() {
    test('Simple.', function() {
        const a = constructors.zeros([2, 2])
        expect(a.shape).toEqual(new Uint32Array([2, 2]))
        expect(a.data).toEqual(new Float64Array(4))
    })
});

describe('ones.', function() {
    test('Simple.', function() {
        const a = constructors.ones([2, 2])
        expect(a.shape).toEqual(new Uint32Array([2, 2]))
        expect(a.data).toEqual(new Float64Array([1, 1, 1, 1]))
    })
});

describe('identity.', function() {
    test('Simple', function() {
        const a = constructors.eye(2);
        for (let [i, j] of a._iorder_index_iterator()) {
            if (i === j) {
                expect(a.g(i, j)).toBe(1);
            } else {
                expect(a.g(i, j)).toBe(0);
            }
        }
    });
});

describe('arange', function() {
    test('Simple', function() {
        const a = constructors.arange(5);
        expect(a.shape).toEqual(new Uint32Array([5]))
        expect(a.data).toEqual(new Int32Array([0, 1, 2, 3, 4]))
    });
})

describe('filled.', function() {
    test('Simple', function() {
        const a = constructors.filled(5, [2, 2]);
        expect(a.shape).toEqual(new Uint32Array([2, 2]));
        expect(a.data).toEqual(new Float64Array([5, 5, 5, 5]))
    });
});

describe("from_nested_array.", function () {

    test('Simple.', function() {
        const nested = constructors.from_nested_array([[0, 1], [2, 3]]);
        expect(nested.shape).toEqual(new Uint32Array([2, 2]));
        expect(nested.data).toEqual(new Float64Array([0, 2, 1, 3]))
    });

    it("hand array.", function () {
        let nested = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]];
        let tensor = constructors.from_nested_array(nested);
        expect(tensor.shape).toEqual(new Uint32Array([2, 2, 2]));
        expect(tensor.g(0, 0, 0)).toBe(0);
        expect(tensor.g(0, 0, 1)).toBe(1);
        expect(tensor.g(0, 1, 0)).toBe(2);
        expect(tensor.g(0, 1, 1)).toBe(3);
        expect(tensor.g(1, 0, 0)).toBe(4);
        expect(tensor.g(1, 0, 1)).toBe(5);
        expect(tensor.g(1, 1, 0)).toBe(6);
        expect(tensor.g(1, 1, 1)).toBe(7);
    });

    it("larger array", function () {
        // Create nested array with dimensions 3 x 2 x 3 x 5
        let nested = [1, 2, 3].map(
        x => [x, x+1].map(
            y => [y, y+1, y+2, y+3].map(
            z => [y + 2, y+3, y+4, y+5, y+6]
            ),
        ),
        );

        let good_nested = constructors.from_nested_array(nested);
        expect(good_nested.shape).toEqual(new Uint32Array([3, 2, 4, 5]));
        for (let indices of good_nested._iorder_index_iterator()) {
        let expected = utils._nested_array_value_from_index(nested, indices);
        let actual = good_nested.g(...indices);
        expect(actual).toBe(expected, `index: ${indices}`);

        expected = nested[indices[0]][indices[1]][indices[2]][indices[3]];
        actual = good_nested.g(...indices);
        expect(actual).toBe(expected, `index: ${indices}`);
        }
        expect(nested[0][0][0][0]).toBe(good_nested.g(0, 0, 0, 0));
        expect(nested[2][0][0][0]).toBe(good_nested.g(2, 0, 0, 0));
    });
});

describe('from_iterable.', function() {
    test('Simple.', function() {
        const array = [1, 2, 3, 4];
        const a = constructors.from_iterable(array, [2, 2]);
        expect(a.shape).toEqual(new Uint32Array([2, 2]));
        expect(a.data).toEqual(new Float64Array([1, 3, 2, 4]))
    });
});

describe('from_json.', function() {
    test('Simple.', function () {
        const json = {
            data: [1, 2, 3, 4],
            dtype: 'float64'
        }
        const a = constructors.from_json(json);
        expect(a.shape).toEqual(new Uint32Array([4]));
        expect(a.data).toEqual(new Float64Array([1, 2, 3, 4]))
    })
})


describe('where', function() {
    test('Simple', function() {
        const cond = constructors.from_nested_array([true, false, true, false, true]);
        const a = constructors.arange(5);
        const b = constructors.arange(6, 11);
        const val = constructors.where(cond, a, b);
        expect(val.shape).toEqual(new Uint32Array([5]));
        expect(val.data).toEqual(new Int32Array([0, 7, 2, 9, 4]));
    });
})
