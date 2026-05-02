from code import config
from code.pii import scrub_text


def test_scrub_email_phone_and_card():
    result = scrub_text(
        "Email me at adi@example.com or call +1 415-555-1212. Card 4111 1111 1111 1111."
    )
    assert config.PII_EMAIL_TOKEN in result.text
    assert config.PII_PHONE_TOKEN in result.text
    assert config.PII_CARD_TOKEN in result.text
    assert result.replacements >= 3


def test_scrub_password_username_and_token_fields():
    result = scrub_text(
        "username: aditya password=supersecret api_key: sk-1234567890abcdef"
    )
    assert f"username: {config.PII_USERNAME_TOKEN}" in result.text
    assert f"password={config.PII_PASSWORD_TOKEN}" in result.text
    assert config.PII_SECRET_TOKEN in result.text


def test_scrub_bearer_and_ip_address():
    result = scrub_text(
        "Authorization: Bearer abcdefghijklmnop and last IP was 192.168.10.42"
    )
    assert "Authorization:" in result.text
    assert config.PII_SECRET_TOKEN in result.text
    assert config.PII_IP_TOKEN in result.text


def test_scrub_multiple_emails_and_formats():
    result = scrub_text(
        "Contact: first.last+test@sub.domain.co.uk and backup mail admin@company.io"
    )
    assert result.text.count(config.PII_EMAIL_TOKEN) == 2

def test_scrub_phone_various_formats():
    result = scrub_text(
        "Numbers: (415)555-1212, 415.555.1212, +91-9876543210"
    )
    assert result.text.count(config.PII_PHONE_TOKEN) >= 3

def test_scrub_card_with_dashes_and_spaces():
    result = scrub_text(
        "Cards: 4111-1111-1111-1111 and 5500 0000 0000 0004"
    )
    assert result.text.count(config.PII_CARD_TOKEN) == 2

def test_scrub_various_secret_patterns():
    result = scrub_text(
        "token=abcd1234 secret: xyz987 apiKey=sk_test_abcdef"
    )
    assert result.text.count(config.PII_SECRET_TOKEN) >= 2

def test_scrub_mixed_pii():
    result = scrub_text(
        "User adi@example.com used card 4111111111111111 from IP 192.168.1.1"
    )
    assert config.PII_EMAIL_TOKEN in result.text
    assert config.PII_CARD_TOKEN in result.text
    assert config.PII_IP_TOKEN in result.text

def test_no_pii_no_change():
    text = "This is a normal support request without sensitive data."
    result = scrub_text(text)
    assert result.text == text
    assert result.replacements == 0

def test_scrub_obfuscated_email():
    result = scrub_text(
        "adi [at] example [dot] com"
    )
    assert result.replacements == 0

def test_partial_card_not_scrubbed():
    result = scrub_text(
        "Last 4 digits: 1234"
    )
    assert config.PII_CARD_TOKEN not in result.text

def test_scrub_json_payload():
    result = scrub_text(
        '{"email":"adi@example.com","password":"secret123"}'
    )
    assert config.PII_EMAIL_TOKEN in result.text
    assert config.PII_PASSWORD_TOKEN in result.text


def test_scrub_url_with_token():
    result = scrub_text(
        "https://api.com?api_key=abcdef123456"
    )
    assert config.PII_SECRET_TOKEN in result.text

def test_large_input():
    text = ("adi@example.com " * 1000)
    result = scrub_text(text)
    assert result.text.count(config.PII_EMAIL_TOKEN) == 1000