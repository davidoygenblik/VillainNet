import itertools

def gen_subnets():
    possible_subnet_settings = [[[3, 3, 3, 3], 2], [[4, 4, 4, 4], 3], [[6, 6, 6, 6], 4]]
    expand_ratio_list = []
    depth_list = []
    all_possible_subnets = itertools.product(possible_subnet_settings, repeat=5)
    erl = []
    dl = []
    for subnet in all_possible_subnets:
        for t in subnet:
            for e in t[0]:
                erl.append(e)
            dl.append(t[1])
        
        if len(erl) == 20:
            expand_ratio_list.append(erl)
            erl = []
        
        if len(dl) == 5:
            depth_list.append(dl)
            dl = []
    return (expand_ratio_list, depth_list)


expand_ratio_list, depth_list = gen_subnets()

print(expand_ratio_list[162])
print(depth_list[162])
