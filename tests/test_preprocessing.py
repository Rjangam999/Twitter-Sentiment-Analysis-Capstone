from src.data.preprocess import clean_text 

def test_clean_text_removes_noise():
    text = "Hello!!! Visit https://test.com @user #awesome 123"
    cleaned = clean_text(text)

    assert "http" not in cleaned 
    assert "@" not in cleaned 
    assert "#" not in cleaned 
    assert "123" not in cleaned 