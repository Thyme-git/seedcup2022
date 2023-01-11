cnt = 0
s = 0
avg = []
with open('./score_logs_vnonrlv2') as f:
    for line in f.readlines():
        if line[0] == '-':
            continue
        cnt += 1
        l = line[1:-2].split(',')
        s += int(l[1])
        # avg.append(s/cnt)


print(s/cnt)