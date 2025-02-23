import pstats

p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(50)  # Print the top 20 functions