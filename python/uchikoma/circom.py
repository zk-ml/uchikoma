from __future__ import annotations

import typing

import os
from os import path
import json
import numpy as np
from dataclasses import dataclass

import re

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

circom_template_string = """
pragma circom 2.1.0;

{include}
template ModelMain() {brace_left}
{signal}
{main}
{brace_right}

component main = ModelMain();
"""

def inject(template, key, code) -> str:
    """ helper function to auto inject sections. """
    # append key to the end
    code += "{{{}}}".format(key)
    return template.format_map(SafeDict({key: code}))

def inject_main(template, code) -> str:
    return inject(template, "main", code + "\n")

def inject_signal(template, code) -> str:
    return inject(template, "signal", code + "\n")

def inject_include(template, code) -> str:
    return inject(template, "include", code + "\n")


class Signal:
    def __init__(self, comp: ComponentInstance, name, shape):
        self.comp = comp
        self.name = name
        self.shape = shape

    def inject(self, code: str, inp: ComponentInstance):
        assert self.shape == inp.shape

        if len(self.shape) == 0:
            circom_assign += "{}.{} <== {};".format(
                    self.comp.name, self.name, inp.circom_output)
            return inject_main(code, circom_assign)

        circom_shape = "{main}"
        for idx, dim in enumerate(self.shape):
            circom_for = (
                "for (var i{idx} = 0; i{idx} < {dim}; i{idx}++) {brace_left}\n"
                "{main}\n"
                "{brace_right}\n"
            ).format_map(SafeDict(idx=idx, dim=dim))
            circom_shape = circom_shape.format_map(
                    SafeDict(main=circom_for.strip()))

        circom_index = ["[i"+str(i)+"]" for i in range(len(self.shape))]
        circom_assign = "\t{}.{}{} <== {}{};".format(
                self.comp.name, self.name, "".join(circom_index),
                inp.circom_output, "".join(circom_index),
                )
        circom_shape = circom_shape.format_map(
                SafeDict(main=circom_assign))
        return inject_main(code, circom_shape)


class CircomGenerator:
    def __init__(self, comp: Component,
            name, inputs: typing.List[CircomGenerator],
            attrs):
        self.comp = comp

        self.name = name
        self.inputs = inputs
        self.attrs = attrs
        assert "shape" in self.attrs, self.info()
        self.shape = self.attrs["shape"]

        self.circom_inputs: typing.List[Signal] = []
        self.circom_args = ""
        self.circom_output = "{}.out".format(self.name)

        self._visit_flag = False

        self.apply()

    def info(self):
        inputs_info = [
            "{}@{}".format(i.name, i.shape) \
            for i in self.inputs ]
        return "{} = {}({}) /* attrs */ \t{}".format(
            self.name, self.comp.op_name,
            ", ".join(inputs_info), self.attrs)

    def apply(self):
        # apply circom inputs, attrs,
        #   output shape, possible output.
        raise NotImplementedError(self.comp.op_name)

    def fill_circom(self, code: str) -> str:
        """ Inject circom code """

        if self._visit_flag:
            return code
        self._visit_flag = True

        # inject inputs circom code
        for inp in self.inputs:
            code = inp.fill_circom(code)

        # inject dependent header
        if self.comp.fpath not in code:
            code = inject_include(code,
                "include \"{}\";".format(self.comp.fpath))

        # component circom code
        circom_component = "component {} = {}({});".format(
                self.name, self.comp.op_name, self.circom_args)
        code = inject_main(code, circom_component)

        return self.fill_inputs(code)

    def fill_inputs(self, code: str):
        assert len(self.circom_inputs) == len(self.inputs)
        for cir_inp, inp in zip(self.circom_inputs, self.inputs):
            code = cir_inp.inject(code, inp)
        return inject_main(code, "")


def generate(comp: ComponentInstance):
    circom_code = comp.fill_circom(circom_template_string)

    # complete circom code
    circom_code = circom_code.format_map(dict(
        include="",
        signal="",
        main="",
        brace_left="{",
        brace_right="}",))
    return circom_code


@dataclass
class Component:
    op_name: str
    fpath: str

    args: typing.Dict[str]

    input_names: typing.List[str]   = None
    input_dims: typing.List[int]    = None
    output_names: typing.List[str]  = None
    output_dims: typing.List[int]   = None

    #  def __init__(self, op_name, args, fpath):
    #      self.op_name = op_name
    #      self.args = [a.strip() for a in args]
    #      self.input_names = []
    #      self.input_dims = []
    #      self.output_names = []
    #      self.output_dims = []
    #      self.fpath = fpath

    def __call__(self, name, inputs, attrs) -> CircomGenerator:
        import zkml.circom_impl
        return eval('zkml.circom_impl.'+self.op_name+'Generator')(
                self, name, inputs, attrs)
        #  return op_components[self.op_name]()
        #  return CircomGenerator(self, name, inputs, attrs)


    def to_string(self):
        args_str = ", ".join(self.args)
        args_str = "(" + args_str + ")"
        return "{:>20}{:30} {}{}{}{} \t<-- {}".format(
            self.op_name, args_str,
            self.input_names, self.input_dims,
            self.output_names, self.output_dims,
            self.fpath)


components: typing.Dict[str, Component] = {
    # pre-defined null component
    "Input": Component("Input", [], "embeded"),
    "Output": Component("Output", [], "embeded")
}

def file_parse(fpath):
    #  print("register circom file: ", fpath)
    with open(fpath, "r") as f:
        lines = f.read().split("\n")

    lines = [l for l in lines if not l.strip().startswith("//")]
    lines = " ".join(lines)

    lines = re.sub("/\*.*?\*/", "IGN", lines)

    funcs = re.findall("template (\w+) ?\((.*?)\) ?\{(.*?)\}", lines)
    for func in funcs:
        op_name = func[0].strip()
        args = func[1].split(",")
        main = func[2].strip()
        assert op_name not in components, \
            "duplicated compoenent: {} in {} vs. {}".format(
                    op_name, components[op_name].fpath, fpath)

        signals = re.findall("signal (\w+) (\w+)(.*?);", main)
        infos = [[] for i in range(4)]
        for sig in signals:
            sig_types = ["input", "output"]
            assert sig[0] in sig_types, sig[1] + " | " + main
            idx = sig_types.index(sig[0])
            infos[idx*2+0].append(sig[1])

            sig_dim = sig[2].count("[")
            infos[idx*2+1].append(sig_dim)
        components[op_name] = Component(
                op_name, fpath,
                [a.strip() for a in args],
                *infos)


def dir_parse(dir_path, skips=[]):
    names = os.listdir(dir_path)
    for name in names:
        if name in skips:
            continue

        fpath = path.join(dir_path, name)
        if os.path.isdir(fpath):
            dir_parse(fpath)
        elif os.path.isfile(fpath):
            if fpath.endswith(".circom"):
                file_parse(fpath)

def info():
    #  print("Circom Operators:", list(components.keys()))
    for comp in components.values():
        if comp.op_name in ["Input", "Output"]:
            continue
        print(comp.to_string())

