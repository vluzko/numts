const call_python = require('../call_python');
const tensor = require('../../numts/tensor').tensor;
const binary_ops = require("../../numts/tensor_core/binary_ops");
const numts = require('../../numts/numts');
const helpers = require('./helpers');


describe('Basic operations.', () => {

  function check_op(op) {
    const f = (a, b) => {
        const numts_value = binary_ops[op](a, b);
        const a_string = JSON.stringify(a.to_json());
        const b_string = JSON.stringify(b.to_json());
        const py_str = call_python.call_python(op, [a_string, b_string])
        const py_value = numts.from_json(JSON.parse(py_str))

        return numts_value.is_close(py_value).all()
    };
    return f;
  }

  test('Array addition.', () => {
    helpers.check_arrays(check_op('_add'));
  });
});
