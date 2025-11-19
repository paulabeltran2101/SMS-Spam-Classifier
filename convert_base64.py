import base64

with open("images/spam_background.png", "rb") as img:
    encoded = base64.b64encode(img.read()).decode("utf-8")

print(encoded)
