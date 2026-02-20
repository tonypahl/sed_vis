from sed_vis import decode_sfh_age

def test_csf_allage():
    assert decode_sfh_age("csf_allage") == ["Constant SFH", "All ages"]

def test_tau_agegt50():
    assert decode_sfh_age("tau_agegt50") == ["Tau model", r"Age $>$ 50Myr"]

def test_unknown_model():
    assert decode_sfh_age("burst_30") == ["burst", "30"]
