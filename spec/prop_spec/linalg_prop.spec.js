const tensor = require("../../numts/tensor").tensor;
const binary_ops = require("../../numts/tensor_core/binary_ops");
const numts = require("../../numts/numts");
const linalg = require("../../numts/linalg");
const helpers = require('./helpers');

describe('Decompositions.', () => {
    describe('LU decomposition', () => {

    })

    describe('Givens QR.', () => {
        test('Thin matrices.', function() {
            function f(a) {
                const [m, ] = a.shape;
                const [q, r] = linalg.qr(a);
    
                // Check that q is orthogonal
                const t = q.transpose();
                const inv_prod = tensor.matmul_2d(q, t);
                const expected = numts.eye(m);
                expect(inv_prod.is_close(expected).all()).toBe(true);
    
                // Check that the product is correct
                const qr_prod = tensor.matmul_2d(q, r);
                expect(qr_prod.is_close(a).all()).toBe(true);
            }
            helpers.check_matrix(f, 'thin');
        });
    })

    describe('Householder QR.', () => {
        test('Thin matrices.', function() {
            function f(a) {
                const [m, ] = a.shape;
                const [q, r] = linalg.qr(a, {algorithm: 'householder'});

                // Check that q is orthogonal
                const t = q.transpose();
                const inv_prod = tensor.matmul_2d(q, t);
                const expected = numts.eye(m);
                if (!inv_prod.is_close(expected).all()) throw new Error('Inverse prod not identity')

                // Check that the product is correct
                const qr_prod = tensor.matmul_2d(q, r);
                if (!qr_prod.is_close(a).all()) throw new Error('qr not original.')

                // Check that R is upper triangular
                const [mr, nr] = r.shape;
                for (let i = 0; i < mr - 1; i++) {
                    const zero_vec = numts.zeros([mr - i - 1], 1);
                    const slice = r.slice([i+1, null], i);
                    if (!zero_vec.is_close(slice)) {throw new Error('R is not upper triangular.')}
                }
            }
            helpers.check_matrix(f, 'thin');
        });

        describe('Householder column vectors.', function() {
            test('Thin matrices.', function() {
                function f(a) {
                    const [m, _] = a.shape;
                    const v = linalg.householder_vector(a, 1, 0);
                    expect(v.shape[0]).toBe(m-1);
                    expect(v.shape[1]).toBe(1);
                }
            })
        })

        describe('Householder transforms.', () => {
            test('Thin matrices.', function() {
                function f(a) {
                    const [m, ] = a.shape;
                    const [q, _] = linalg.householder_col_vector(a, m-2, 0);

                    const col_slice = q.slice([m-2, null], 0);
                    const expected = numts.zeros([m-2, 1]);
                    if (!(col_slice.is_close(expected).all())) throw new Error('not zeroed')
                    // // Check that q is orthogonal
                    // const t = q.transpose();
                    // const inv_prod = tensor.matmul_2d(q, t);
                    // const expected = numts.eye(m);
                    // if (!inv_prod.is_close(expected).all()) throw new Error('Inverse prod not identity')

                    // // Check that the product is correct
                    // const qr_prod = tensor.matmul_2d(q, r);
                    // if (!qr_prod.is_close(a).all()) throw new Error('qr not original.')
                }
                helpers.check_matrix(f, 'thin');
            })
        })
    })
})
