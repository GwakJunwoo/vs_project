import pandas as pd

weight1 = 0.0
weight2 = 0.0
bias = 0.0

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

for test_inputs, correct_outputs in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_inputs[0] + weight2 * test_inputs[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_outputs else 'No'
    outputs.append([test_inputs[0], test_inputs[1], linear_combination, output, is_correct_string])


# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
    char = list()
    value = 0
    for i in range(len(s) - k + 1):
        for j in range(k):
            char.append(s[j])
        for n, m in enumerate(char):
            value += (1 + ord(m) - ord('a')) * power ^ n
        if value == hashValue:
            return "".join(char)

class Solution:
    def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        i = -1
        while True:
            i += 1
            char = [s[i+j] for j in range(k)]
            value = sum((1+ord(m)-ord('a'))*(power**n) for n, m in enumerate(char))

            value = value%modulo
            if value == hashValue:
                return "".join(char)