| map | set | game | oracle | frozen: won/10, mean eff | passive: actions (eff) | active: actions (eff), experiments | active's rule | memory carried | missing variable |
|---|---|---|---|---|---|---|---|---|---|
| small | dev | occupancy | 14 | 7/10, 0.17 | won 14 (1.00) | won 17 (0.68), 4 | onpad | onpad | no |
| small | dev | toggle | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | entries odd | entries | no |
| small | dev | timer4 | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | since_entry<4 | since entry | no |
| small | seen after v1 (not held out) | second_entry | 10 | 5/10, 0.09 | won 16 (0.39) | won 10 (1.00), 3 | entries>=2 | entries | no |
| small | seen after v1 (not held out) | delay3 | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | since_entry<4 | since entry | no |
| small | seen after v1 (not held out) | latch | 8 | 7/10, 0.10 | won 14 (0.33) | won 8 (1.00), 2 | entries odd | entries | no |
| small | seen after v1 (not held out) | clock | 6 | 10/10, 0.17 | won 6 (1.00) | won 6 (1.00), 0 | t even | t mod 2 | no |
| small | seen after v1 (not held out) | last_move_L | 8 | 10/10, 0.08 | LOST 12 (0.00) | LOST 12 (0.00), 3 | None |  | yes |
| small | held out (after v2 freeze) | inverse | 6 | 10/10, 1.00 | won 6 (1.00) | won 6 (1.00), 0 | not onpad | onpad | no |
| small | held out (after v2 freeze) | brief_second | 10 | 5/10, 0.09 | won 16 (0.39) | won 10 (1.00), 3 | entries>=2 | entries | no |
| small | held out (after v2 freeze) | box_only | 14 | 0/10, 0.00 | won 14 (1.00) | LOST 120 (0.00), 3 | onpad and since_exit<1 | onpad, since exit | no |
| small | held out (after v2 freeze) | exits_odd | 8 | 7/10, 0.08 | won 14 (0.33) | won 8 (1.00), 2 | since_exit<4 | since exit | no |
| corridor | held out map | occupancy | 28 | 3/10, 0.07 | won 32 (0.77) | won 49 (0.33), 12 | onpad | onpad | no |
| corridor | held out map | toggle | 24 | 3/10, 0.07 | won 24 (1.00) | won 38 (0.40), 14 | entries odd | entries | no |
| corridor | held out map | timer14 | 24 | 3/10, 0.07 | won 24 (1.00) | won 42 (0.33), 18 | since_entry<14 | since entry | no |
| corridor | held out map | latch | 24 | 3/10, 0.07 | won 24 (1.00) | won 40 (0.36), 18 | entries>=1 | entries | no |
| corridor | held out map | inverse | 2 | 10/10, 1.00 | won 2 (1.00) | won 2 (1.00), 0 | not onpad | onpad | no |
| corridor | held out map | brief_second | unsolvable | | | | | | |
| corridor | held out map | box_only | 28 | 0/10, 0.00 | LOST 21 (0.00) | LOST 120 (0.00), 7 | onpad and since_exit<1 | onpad, since exit | no |
