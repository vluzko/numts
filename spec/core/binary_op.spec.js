const numts = require('../../numts/numts');
const binary_ops = require('../../numts/tensor_core/binary_ops');
const call_python = require('../call_python');
const tensor = numts.tensor;

describe('binary_broadcast.', function () {
    test('return first.', function () {
        let a = numts.arange(0, 10);
        let b = numts.arange(1);
        let f = (a, b) => a;

        let broadcasted = binary_ops._binary_broadcast(a, b, f);
        expect(broadcasted.equals(a)).toBe(true);
    });

    test('return second.', function () {
        let a = numts.arange(1);
        let b = numts.arange(0, 10);
        let f = (a, b) => b;

        let broadcasted = binary_ops._binary_broadcast(a, b, f);
        expect(broadcasted.equals(b)).toBe(true);
    });

    test('take_max', function () {
        const a = numts.arange(30).reshape(5, 6);
        const b = numts.arange(30, 60).reshape(5, 6);
        const c = tensor.take_max(a, b);
        expect(b.equals(c)).toBe(true);
    });

    test('take_min', function () {
        const a = numts.arange(30).reshape(5, 6);
        const b = numts.arange(30, 60).reshape(5, 6);
        const c = tensor.take_min(a, b);
        expect(a.equals(c)).toBe(true);
    });
});

describe('arithmetic.', function () {
    const a = numts.from_nested_array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], 'int32');
    const b = numts.arange(1, 6);
    test('add.', function () {
        let summed = binary_ops._add(a, b);
        let expected = numts.from_nested_array([[1, 3, 5, 7, 9], [6, 8, 10, 12, 14]], 'int32');
        expect(summed.equals(expected)).toBe(true);
    });

    test('sub.', function () {
        let sub = binary_ops._sub(a, b);
        let expected = numts.from_nested_array([[-1, -1, -1, -1, -1], [4, 4, 4, 4, 4]], 'int32');
        expect(sub.equals(expected)).toBe(true);
    });

    test('mult.', function () {
        let product = binary_ops._mult(a, b);
        let expected = numts.from_nested_array([[0, 2, 6, 12, 20], [5, 12, 21, 32, 45]], 'int32');
        expect(product.equals(expected)).toBe(true);
    });

    test('div.', function () {
        let product = binary_ops._div(a, b);
        let expected = numts.from_nested_array([
            [0, 1 / 2, 2 / 3, 3 / 4, 4 / 5],
            [5, 6 / 2, 7 / 3, 8 / 4, 9 / 5]], 'float64');
        expect(product.equals(expected)).toBe(true);
    });

    test('mod.', function () {
        let product = binary_ops._mod(a, b);
        let expected = numts.from_nested_array([[0, 1, 2, 3, 4], [0, 0, 1, 0, 4]], 'int32');
        expect(product.equals(expected)).toBe(true);
    });

    test('fdiv.', function () {
        let product = binary_ops._fdiv(a, b);
        let expected = numts.from_nested_array([[0, 0, 0, 0, 0], [5, 3, 2, 2, 1]], 'int32');
        expect(product.equals(expected)).toBe(true);
    });

    test('cdiv.', function () {
        let product = binary_ops._cdiv(a, b);
        let expected = numts.from_nested_array([[0, 1, 1, 1, 1], [5, 3, 3, 2, 2]], 'int32');
        expect(product.equals(expected)).toBe(true);
    });

    describe('Regression failures.', function () {

        test('_add failure. 2021-08-21', function () {
            const shape = [1, 4];
            const a = numts.from_iterable([0, 4.172325134277344e-7, 0.47082263231277466, 5.960464477539062e-7], shape);
            const b = numts.from_iterable([0, 4.172325134277344e-7, 0.47082263231277466, 5.960464477539062e-7], shape);

            const numts_value = binary_ops._add(a, b);

            const a_string = JSON.stringify(a.to_json());
            const b_string = JSON.stringify(b.to_json());
            const py_str = call_python.call_python('_add', [a_string, b_string])
            const py_value = numts.from_json(JSON.parse(py_str))

            const result = numts_value.is_close(py_value)
            expect(result.all()).toBe(true);
        })
    })
});

describe('boolean.', function () {

})

describe('matmul_2d.', function () {
    test('scalar.', function () {
        let a = numts.arange(1, 2).reshape([1, 1]);
        let b = numts.arange(10, 11).reshape([1, 1]);
        let x = tensor.matmul_2d(a, b);
        const expected = numts.from_nested_array([[10]]);
        expect(expected.equals(x)).toBe(true);
    });

    test('2d test', function () {
        const a = numts.arange(4).reshape(2, 2);
        const b = numts.arange(4, 8).reshape(2, 2);
        const expected = numts.from_nested_array([
            [6, 7],
            [26, 31]
        ]);

        const actual = tensor.matmul_2d(a, b);
        expect(actual.equals(expected)).toBe(true);

    });

    describe('from breaks.', function () {
        test('broken broadcast test.', function () {
            const a = numts.arange(24).reshape(2, 3, 4).slice(0);
            const b = numts.arange(16).reshape(4, 4);
            const expected = numts.from_nested_array([
                [56, 62, 68, 74],
                [152, 174, 196, 218],
                [248, 286, 324, 362]
            ]);

            const actual = tensor.matmul_2d(a, b);
            expect(expected.equals(actual)).toBe(true);
        });

    });
});

describe('matmul.', function () {

    test('scalar.', function () {
        let a = numts.arange(1, 2).reshape([1, 1]);
        let b = numts.arange(10, 11).reshape([1, 1]);
        let x = binary_ops.broadcast_matmul(a, b);
        const expected = numts.from_nested_array([[10]]);
        expect(expected.equals(x)).toBe(true);
    });

    test('m x n by n by k', function () {
        let a = numts.arange(15).reshape([5, 3]);
        let b = numts.arange(12).reshape(3, 4);

        let x = binary_ops.broadcast_matmul(a, b);
        const expected = numts.from_nested_array([

        ]);
        expect(expected.equals(x));
    });

    test('broadcast test.', function () {
        let a = numts.arange(24).reshape([2, 3, 4]);
        let b = numts.arange(16).reshape(4, 4);

        let x = binary_ops.broadcast_matmul(a, b);
        const expected = numts.from_nested_array([
            [[56, 62, 68, 74],
            [152, 174, 196, 218],
            [248, 286, 324, 362]],

            [[344, 398, 452, 506],
            [440, 510, 580, 650],
            [536, 622, 708, 794]]
        ], 'int32');
        expect(expected.equals(x)).toBe(true);
    });
});

describe('dot.', function () {
    test('simple.', function () {
        const a = numts.arange(10, 20);
        const b = numts.arange(20, 30);
        const dot = tensor.dot(a, b);
        expect(dot).toBe(3635)
    });
});


describe('tensordot.', function () {
    test('simple.', function () {
        const a = numts.arange(60).reshape(3,4,5);
        const b = numts.arange(24).reshape(4,3,2);
        const c = binary_ops.tensordot(a, b, [[1, 0], [0, 1]]);
        expect(c.shape).toEqual(new Uint32Array([5, 2]));
    });

    describe('dot_product.', function () {

        // Should be equivalent to a dot product
        test('one_dim_axes_1.', function () {
            const a = numts.arange(10);
            const b = numts.arange(10);
            const c = binary_ops.tensordot(a, b, 1);
            expect(c.shape).toEqual(new Uint32Array([1]));
            expect(c.data).toEqual(new Int32Array([285]));
        });

        test('one_dimensional_b', function() {
            const a = numts.arange(10).reshape(2, 5);
            const b = numts.arange(5);
            const c = binary_ops.tensordot(a, b, 1);
            expect(c.shape).toEqual(new Uint32Array([1]));
            expect(c.data).toEqual(new Int32Array([285]));
        });

        test('one_dimensional_a', function() {
            const a = numts.arange(5);
            const b = numts.arange(10).reshape(5, 2);
            const c = binary_ops.tensordot(a, b, 1);
            expect(c.shape).toEqual(new Uint32Array([1]));
            expect(c.data).toEqual(new Int32Array([285]));
        });

        // Should be equivalent to matrix multiplication
        test('one_dim_axes_2.', function () {
            const a = numts.arange(4).reshape(2, 2);
            const c = binary_ops.tensordot(a, a, 1);
            expect(c.shape).toEqual(new Uint32Array([2, 2]));
            expect(c.data).toEqual(new Int32Array([2, 6, 3, 11]));
        });
    });
});