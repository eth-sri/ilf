import json
from .opcode import STACK_CHANGES


class Top:

    def __init__(self):
        pass


    def copy(self):
        return Top()


    def __hash__(self):
        return hash('Top')


    def __eq__(self, other):
        return isinstance(other, Top)


    def __str__(self):
        return 'Top'


    def __repr__(self):
        return str(self)


class TopCALLDATASIZE(Top):

    def __init__(self):
        super().__init__()


    def copy(self):
        return TopCALLDATASIZE()


    def __hash__(self):
        return hash('TopCALLDATASIZE')


    def __eq__(self, other):
        return isinstance(other, TopCALLDATASIZE)


    def __str__(self):
        return 'TopCALLDATASIZE'


class Value:

    def __init__(self, value):
        self.value = value
        assert not is_top(value), 'value inside Value is top'
        assert not value.__class__ == Value, 'value inside Value is value'


    def copy(self):
        return Value(self.value)


    def __eq__(self, other):
        return isinstance(other, Value) and self.value == other.value


    def __hash__(self):
        return hash(('Value', self.value))


    def __str__(self):
        return '0x{:x}'.format(self.value)


    def __repr__(self):
        return str(self)


def is_top(elem):
    return isinstance(elem, Top)


def all_not_top(elems):
    return all([not is_top(elem) for elem in elems])


class StackChecker:

    def __init__(self, insn, state):
        self.insn = insn
        self.state = state


    def __enter__(self):
        self.old_len = len(self.state.stack)


    def __exit__(self, *args):
        new_len = len(self.state.stack)
        assert new_len - self.old_len == STACK_CHANGES[self.insn.op], 'stack change is wrong for insn {}'.format(str(self.insn))


class EVMState:

    def __init__(self):
        self.stack = list()
        self.mem = list()
        self.mem_len_decidable = True
        self.storage = dict()

        self.block_trace = list()


    def __hash__(self):
        t = tuple([tuple(self.stack), tuple(self.mem), self.mem_len_decidable, tuple(self.storage.items())])
        return hash(t)


    def __eq__(self, other):
        return self.__class__ == other.__class__ \
            and self.stack == other.stack \
            and self.mem == other.mem \
            and self.mem_len_decidable == other.mem_len_decidable \
            and self.storage == other.storage


    def copy(self):
        cpy = EVMState()
        cpy.stack = self.stack.copy()
        cpy.mem = self.mem.copy()
        cpy.mem_len_decidable = self.mem_len_decidable
        cpy.storage = self.storage.copy()
        cpy.block_trace = self.block_trace.copy()
        return cpy


    def mem_extend(self, length):
        if length > len(self.mem):
            self.mem += [Top() for _ in range(length - len(self.mem))]


    def mem_reset(self, offset, length):
        if len(self.mem) < offset + length:
            self.mem_extend(offset + length)

        for i in range(offset, offset + length):
            self.mem[i] = Top()


    def mem_reset_all(self):
        self.mem.clear()
        self.mem_len_decidable = False


    def mem_load(self, offset, length):
        bs = bytearray(map(lambda m: m.value, self.mem[offset:offset + length]))
        return int.from_bytes(bs, byteorder='big')


    def mem_valid(self, offset, length):
        if len(self.mem) < offset + length:
            return False
        else:
            return all_not_top(self.mem[offset:offset+length])


    def mem_store(self, offset, values):
        if len(self.mem) < len(values) + offset:
            self.mem_extend(len(values) + offset)

        for i in range(len(values)):
            self.mem[offset + i] = Value(values[i])


    def mem_store_top(self, offset, length):
        if len(self.mem) < offset + length:
            self.mem_extend(offset + length)

        for i in range(offset, offset + length):
            self.mem[i] = Top()


    def get_mem_intro_idx(self, offset, length):
        mem_entries = self.mem[offset:offset+length]
        intro_idx_set = set(map(lambda e: e.intro_idx, mem_entries))
        if len(intro_idx_set) == 1:
            return list(intro_idx_set)[0]
        else:
            return -1


    def mem_size(self):
        if self.mem_len_decidable:
            return Value(len(self.mem))
        else:
            return Top()


    def pop_stack(self, size=1):
        if size == 1:
            s = self.stack.pop()
            if not is_top(s):
                s = s.value
            return s
        elif size > 1:
            vals = []
            for _ in range(size):
                s = self.stack.pop()
                if not is_top(s):
                    s = s.value
                vals.append(s)
            return tuple(vals)
        else:
            assert False, 'cannot pop non positive number of elements from stack'


    def push_stack(self, value):
        self.stack.append(value)


    def __str__(self):
        j = {
            'statck': str(self.stack),
            'mem': str(self.mem),
            'storage': str(self.storage),
            'mem_len_decidable': self.mem_len_decidable,
            'block_trace': self.block_trace,
        }
        return json.dumps(j)