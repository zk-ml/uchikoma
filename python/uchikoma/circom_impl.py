
from .circom import *


"""
    Generator Implementation

    override apply function, several variables to be set:

        1. circom_input
        2. circom_args
        3. circom_output
        4. shape 
"""

class InputGenerator(CircomGenerator):
    def apply(self):
        self.circom_output = self.name

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        circom_shape = [str(s) for s in self.shape ]
        circom_shape = ["["+s+"]" for s in circom_shape]
        return inject_signal(code, "signal input {}{};".format(
                self.name, "".join(circom_shape)))


class OutputGenerator(CircomGenerator):
    def apply(self):
        pass

    def fill_circom(self, code: str) -> str:
        if self._visit_flag:
            return code
        self._visit_flag = True

        assert len(self.inputs) == 1
        assert self.shape == self.inputs[0].shape

        for inp in self.inputs:
            code = inp.fill_circom(code)

        circom_shape = ["["+str(s)+"]" for s in self.shape]
        code = inject_signal(code, "signal output {}{};".format(
                    self.name, "".join(circom_shape)))

        circom_shape = "{main}"
        for idx, dim in enumerate(self.shape):
            circom_for = (
                "for (var i{idx} = 0; i{idx} < {dim}; i{idx}++) {brace_left}\n"
                "{main}\n"
                "{brace_right}\n"
            ).format_map(SafeDict(idx=idx, dim=dim))
            circom_shape = circom_shape.format_map(
                    SafeDict(main=circom_for.strip()))

        circom_index = ["[i"+str(i)+"]" \
                for i in range(len(self.shape))]
        circom_assign = "\t{}{} <== {}{};".format(
                self.name, "".join(circom_index),
                self.inputs[0].circom_output, "".join(circom_index),
                )
        circom_shape = circom_shape.format_map(
                SafeDict(main=circom_assign))
        return inject_main(code, circom_shape)

class OperatorGenerator(CircomGenerator):
    def apply(self):
        input_shapes = [inp.shape for inp in self.inputs]
        # check input shape dimensions match
        #  print(self.comp.input_dims, input_shapes, self.info())
        assert len(self.comp.input_names) == len(self.inputs)
        for shape in zip(self.comp.input_dims, input_shapes):
            assert shape[0] == len(shape[1]), (
                "{}({}) shape dim not matched, "
                "{} vs. {}, maybe apply shape-adaptor pass."
            ).format(self.name, self.comp.op_name,
                    shape[0], len(shape[1]))

        self.circom_inputs = [
                Signal(self, *info) for info in zip(
                    self.comp.input_names, input_shapes) ]

        args = self.arguments()
        # all arguments must be integers.
        assert all([isinstance(a, int) for a in args]), self.info()
        self.circom_args = ", ".join([
            str(s) for s in self.arguments()])

        #  self.circom_output = self.output_name()
        assert len(self.comp.output_names) == 1
        self.circom_output = "{}.{}".format(
                self.name, self.comp.output_names[0])

        # check output shape dimensions match
        assert self.comp.output_dims[0] == len(self.shape), self.info()

    def arguments(self):
        raise NotImplementedError(self.comp.op_name)

class ShapeGenerator(OperatorGenerator):
    def arguments(self):
        return [ *self.shape ]
class Broadcast3DAxis0SubGenerator(ShapeGenerator):
    pass
class Broadcast3DAxis0AddGenerator(ShapeGenerator):
    pass
class Element1DAddGenerator(ShapeGenerator):
    pass
class Element3DAddGenerator(ShapeGenerator):
    pass
class Element1DSubGenerator(ShapeGenerator):
    pass
class Element1DMulGenerator(ShapeGenerator):
    pass

#  class ElementGenerator(OperatorGenerator):
#      def arguments(self):
#          return [ self.shape[0], ]
#  class ElementAddGenerator(ElementGenerator):
#      pass
#  class ElementSubGenerator(ElementGenerator):
#      pass
#  class ElementMulGenerator(ElementGenerator):
#      pass

class Conv2D_CHWGenerator(OperatorGenerator):
    def arguments(self):
        padding = self.attrs["padding"]
        assert all([p == 0 for p in padding])

        filters = self.attrs["channels"]
        kernel_size = self.attrs["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
        return [ *self.inputs[0].shape, filters, kernel_size, 1, ]

class Pad2DGenerator(OperatorGenerator):
    def arguments(self):
        pad_value = self.attrs.get("scalar", None)
        if pad_value is None:
            pad_value = self.attrs["pad_value"]
        pad_width = [[p, p] if isinstance(p, int) else p \
                for p in self.attrs["pad_width"]]

        for pt in pad_width[:-2]:
            assert pt[0] == 0
            assert pt[1] == 0
        pad_width = [i for p in pad_width[-2:] for i in p]

        return [ *self.inputs[0].shape, pad_value, *pad_width ]

class Resize2DGenerator(OperatorGenerator):
    def arguments(self):
        method = self.attrs.get("method", "nearest_neighbor")
        assert method == "nearest_neighbor"

        input_shape = self.inputs[0].shape
        scaleX = self.shape[1] / input_shape[1]
        scaleY = self.shape[2] / input_shape[2]
        assert scaleX == scaleY
        assert int(scaleX) == scaleX
        return self.inputs[0].shape + [ int(scaleX), ]


def reshape_validate(shape_one, shape_arr, msg):
        assert len(shape_one) == 1
        total_len = 1
        for s in shape_arr:
            total_len *= s
        assert shape_one[0] == total_len, msg

class ReShapeGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.inputs[0].shape,
                self.shape, self.info())
        return self.shape
class ReShape2DGenerator(ReShapeGenerator):
    pass
class ReShape3DGenerator(ReShapeGenerator):
    pass
class ReShape4DGenerator(ReShapeGenerator):
    pass

class FlattenGenerator(OperatorGenerator):
    def arguments(self):
        reshape_validate(
                self.shape,
                self.inputs[0].shape, self.info())
        return self.inputs[0].attrs["shape"]
class Flatten2DGenerator(FlattenGenerator):
    pass
class Flatten3DGenerator(FlattenGenerator):
    pass
class Flatten4DGenerator(FlattenGenerator):
    pass

class Dense2Generator(OperatorGenerator):
    def arguments(self):
        return self.inputs[1].shape

class ScalarGenerator(OperatorGenerator):
    def arguments(self):
        ishape = self.inputs[0].shape
        assert len(ishape) == 1
        return [ishape[0], self.attrs["scalar"]]
class MulScalarGenerator(ScalarGenerator):
    pass
class AddScalarGenerator(ScalarGenerator):
    pass
class SubScalarGenerator(ScalarGenerator):
    pass

class RightShiftGenerator(OperatorGenerator):
    def arguments(self):
        return [ self.shape[0], self.attrs["shift_bit"], ]


class ClipGenerator(OperatorGenerator):
    def arguments(self):
        return [ self.shape[0],
                self.attrs["a_min"], self.attrs["a_max"] ]


