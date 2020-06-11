def func_default(arg1=1, arg2=2, arg3=3):
    print(arg1)
    print(arg2)
    print(arg3)

none_dict = {}
print('dict type is {}'.format(type(none_dict)))
print(isinstance(none_dict, dict))
func_default(4, arg2=2, **none_dict)