file1 = open('log_pointnet_net_512.txt', 'r')
Lines = file1.readlines()
  
count = 0.
times = 0.
# Strips the newline character
for line in Lines:
    count += 1
    times += float(line.strip())

times = times/ count

print(times)
    