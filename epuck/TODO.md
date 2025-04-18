## Notes for 25-0418
- [ ] Check that acceleration is in correct axis
- [ ] Follower and Leader positions need to start at 0 and -0.1
- [ ] Time gap is extremely large, does follower move or approach speed of leader at steady set speed?
   - [ ] Make sure to tune gains for both PID (in `experiment_controller.py` and the backstepping algorithm `epuck-exp.py`