class Value:
    def __init__(self, value, left=None, right=None, _op=''):
        self.value = value
        self.left = left
        self.right = right
        self._op = _op
        self.grad = 0

    def __repr__(self):
        return f'Value({self.value})'

    def __add__(self, other):
        return Value(self.value + other.value, self, other, '+')

    def __mul__(self, other):
        return Value(self.value * other.value, self, other, '*')


# Recursive
def back1(node: Value):
    if not node.left:
        assert not node.right
        return

    if node._op == '+':
        node.left.grad = node.grad
        node.right.grad = node.grad

    elif node._op == '*':
        node.left.grad = node.grad * node.right.value
        node.right.grad = node.grad * node.left.value
    else:
        raise ValueError("Undefined operarand")

    back1(node.left)
    back1(node.right)


# Iterative
def back2(root: Value):
    stack = [root]
    while stack:
        curr = stack.pop()
        if not curr.left:
            assert not curr.right
            continue

        if curr._op == '+':
            curr.left.grad = curr.grad
            curr.right.grad = curr.grad

        elif curr._op == '*':
            curr.left.grad = curr.grad * curr.right.value
            curr.right.grad = curr.grad * curr.left.value
        else:
            raise ValueError("Undefined operarand")

        stack.append(curr.left)
        stack.append(curr.right)


def step(params: list[Value], rate=0.1):
    for p in params:
        p.value -= p.grad * rate
