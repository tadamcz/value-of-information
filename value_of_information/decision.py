def no_signal(prior_ev, bar):
	if prior_ev > bar:
		return {"decision": "d_2"}
	else:
		return {"decision": "d_1"}