const tensor = require('../numts/tensor').tensor;
const numts = require('../numts/numts');
const linalg = require('../numts/linalg');

describe('Matrix norms.', function() {

    describe('l2.', () => {
        it('Simple.', () => {
        const a = numts.arange(25);
        const b = linalg.l2(a);
        expect(b).toBe(70);
        });
    });

    describe('l1', () => {

    });
});

describe('Decompositions.', () =>  {
    describe('LU decomposition.', () => {
        it('No pivoting.', () =>  {
            const a = numts.from_nested_array([
                [4, 2, -1, 3],
                [3, -4, 2, 5],
                [-2, 6, -5, -2],
                [5, 1, 6, -3]
            ]);
            const [l, u] = linalg.lu(a);
            const exp_l = numts.from_nested_array([
                [1, 0, 0, 0],
                [0.75, 1, 0, 0],
                [-0.5, -14 / 11, 1, 0],
                [1.25, 3/11, -13/4, 1]
            ]);
            const exp_u = numts.from_nested_array([
                [4, 2, -1, 3],
                [0, -11/2, 11/4, 11/4],
                [0, 0, -2, 3],
                [0, 0, 0, 9/4]
            ])
            expect(exp_l.is_close(l));
            expect(exp_u.is_close(u));
        });
    });

    describe('QR decomposition.', () =>  {
        describe('Givens QR.' , () =>  {
            it('Basic test.', () =>  {
                const a = numts.arange(15).reshape(5, 3);
                const [q, r] = linalg.qr(a);
                const prod = tensor.matmul_2d(q, r);

                expect(a.is_close(prod).all()).toBe(true);
            });
        });

        describe('Householder QR', () => {
            it('Basic test.', () =>  {
            const a = numts.from_nested_array([
                [1, 6,  11],
                [2, 7, 12],
                [3, 8, 13],
                [4, 9, 14],
                [5, 10, 15]
            ])
            const [m, ] = a.shape;
            const [q, r] = linalg.qr(a, {algorithm: 'householder'});
            const prod = tensor.matmul_2d(q, r);

            const inv_prod = tensor.matmul_2d(q, q.transpose());
            const expected = numts.eye(m);
            expect(inv_prod.is_close(expected).all()).toBe(true);
            expect(a.is_close(prod).all()).toBe(true);
        });

        describe('Householder transformations', () => {
            it('Basic test.', () => {
                const a = numts.from_nested_array([
                    [1, 6,  11],
                    [2, 7, 12],
                    [3, 8, 13],
                    [4, 9, 14],
                    [5, 10, 15]
                ]);
                const [q, _] = linalg.householder_col_vector(a, 3, 0);
                expect(q.shape.length).toBe(2);
                const [i, j] = q.shape;
                expect(i).toBe(5 - 3);
                expect(j).toBe(1);
            });

        })

            describe('Householder vector and matrix.', () => {
                it('Column test.', () => {
                    const a = numts.from_nested_array([
                        [1, 6,  11],
                        [2, 7, 12],
                        [3, 8, 13],
                        [4, 9, 14],
                        [5, 10, 15]
                    ]);
                    const [v, b] = linalg.householder_col_vector(a, 0, 0);
                    expect(v.shape[0]).toBe(5);
                    expect(v.shape[1]).toBe(1)
                    const q = linalg.full_h_col_matrix(v, 5, b);
                    const prod = tensor.matmul_2d(q, a);
                    const close = prod.slice([2, null], 0).is_close(numts.zeros(3));
                    expect(close.all()).toBe(true);
                })

                it('Row test.', () => {
                    const a = numts.from_nested_array([
                        [1, 6,  11, 16],
                        [2, 7, 12, 17],
                        [3, 8, 13, 18],
                        [4, 9, 14, 19],
                        [5, 10, 15, 20]
                    ]);
                    const [v, b] = linalg.householder_row_vector(a, 0, 1);
                    expect(v.shape[0]).toBe(1);
                    expect(v.shape[1]).toBe(3);

                    const q = linalg.full_h_row_matrix(v, 4, b);
                    const prod = tensor.matmul_2d(a, q);

                    const close = prod.slice(0, [2, null]).is_close(numts.zeros(2));
                    expect(close.all()).toBe(true);
                })

                it('Row vs transpose test.', () => {
                    const a = numts.from_nested_array([
                        [1, 6,  11, 16],
                        [2, 7, 12, 17],
                        [3, 8, 13, 18],
                        [4, 9, 14, 19],
                        [5, 10, 15, 20]
                    ]);
                    const [v1, b1] = linalg.householder_row_vector(a, 0, 1);
                    const [v2, b2] = linalg.householder_col_vector(a.transpose(), 1, 0);
                    const q1 = linalg.full_h_row_matrix(v1, 4, b1);
                    const q2 = linalg.full_h_col_matrix(v2, 4, b2);
                    expect(q1.is_close(q2).all()).toBe(true);
                })
            })

            describe('Householder bidiagonal', () => {

                test('Basic test.', () => {
                    const a = numts.from_nested_array([
                        [1, 6,  11, 16],
                        [2, 7, 12, 17],
                        [3, 8, 13, 0.5],
                        [4, 9, 14, 0.1],
                        [5, 10, 15, 100]
                    ]);
                    let [u, s, v] = linalg.householder_bidiagonal(a);
                    const b = tensor.matmul_2d(u, tensor.matmul_2d(s, v.transpose()));
                    expect(a.is_close(b).all()).toBe(true);

                    // Check that s is bidiagonal.
                    expect(linalg.check_bidiagonal(s)).toBe(true);

                    // Check orthogonality of U and V.
                    expect(linalg.check_orthogonal(u)).toBe(true);
                    expect(linalg.check_orthogonal(v)).toBe(true);
                })
            })

            describe('From failures.', () =>  {
                // Failure doesn't repeat.
                it('1. Generated by fast-check.', () =>  {
                const a = numts.from_nested_array([
                    [0, 0.03880476951599121],
                    [0.9937839508056641, 0.5671613216400146]
                ]);
                const [q, r] = linalg.qr(a, {algorithm: 'householder'});
                
                const inv_prod = tensor.matmul_2d(q, q.transpose());
                expect(inv_prod.is_close(numts.eye(2)).all()).toBe(true);
                const qr_prod = tensor.matmul_2d(q, r);
                expect(qr_prod.is_close(a).all()).toBe(true);
                });

                // Failure doesn't repeat.
                it('2. Generated by fast-check.', () =>  {
                const a = numts.from_nested_array([
                    [0, 0.9712722897529602],
                    [0.7647293210029602, 0.32188379764556885],
                    [0.3959425091743469, 0.7986384630203247]
                ]);
                const [q, r] = linalg.qr(a, {algorithm: 'householder'});
                
                const inv_prod = tensor.matmul_2d(q, q.transpose());
                expect(inv_prod.is_close(numts.eye(3)).all()).toBe(true);
                const qr_prod = tensor.matmul_2d(q, r);
                expect(qr_prod.is_close(a).all()).toBe(true);
                });
            });
        });
    });

    describe('Singular value decomposition.', function() {
        test('Basic test.', () => {
            const a = numts.from_nested_array([
                [1, 2, 3, 4],
                [3.5, 2.7, -1, 2],
                [0.2, 6.7, 10, 15],
                [-3.6, 2.6, 1, -1]
            ]);
            // console.log(a.to_nested_array())
            const [u, s, v] = linalg.svd(a);
            console.log(s.to_nested_array());
        });
    });
});

describe('Helper functions.', () => {
    describe('Givens rotations.', () => {
        test('Column test.', () => {
            const a = numts.from_nested_array([
                [1, 0, 0],
                [2, 1, 0],
                [0, 0, 1]
            ]);
            const [_, r] = linalg.givens_rotation_up(a, 0, 1);
            expect(r.g(1, 0)).toBe(0);
        })

        test('Row test.', () => {
            const a = numts.from_nested_array([
                [1, 2, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]);
            const [g, r] = linalg.givens_rotation_row(a, 0, 1);
            expect(r.g(0, 1)).toBe(0);
        })

        describe('Givens values.', () => {
            test('Basic tests.', () => {
                let [c, s] = linalg.givens_values(2, 2);

                expect(Math.abs(c - 0.707106)).toBeLessThan(0.0001);
                expect(Math.abs(s - -0.707106)).toBeLessThan(0.0001);

                [c, s] = linalg.givens_values(2, 2);
                expect(Math.abs(c - 0.707106)).toBeLessThan(0.0001);
                expect(Math.abs(s - -0.707106)).toBeLessThan(0.0001);
            })
        })
    });
})
