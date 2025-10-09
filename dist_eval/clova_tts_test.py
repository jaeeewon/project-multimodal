import requests

url = "https://papago.naver.com/apis/tts/makeID"
clova_host = "https://papago.naver.com/apis/tts/{}"

form_data = {
    "alpha": 0,
    "pitch": 0,
    "speaker": "clara",
    "speed": 0,
    "text": "For pt.III. see Prikl. Mat. Informatika, MAKS Press, no. 4, p. 5-56 (2000). This is a survey of the literature on hybrid simulation of the Kelvin-Helmholtz instability. We start with a brief review of the theory: the simplest model of the instability - a transition layer in the form of a tangential discontinuity; compressibility of the medium; finite size of the velocity shear region; pressure anisotropy. We then describe the electromagnetic hybrid model (ions as particles and electrons as a massless fluid) and the main numerical schemes. We review the studies on two-dimensional and three-dimensional hybrid simulation of the process of particle mixing across the magnetopause shear layer driven by the onset of a Kelvin-Helmholtz instability. The article concludes with a survey of literature on hybrid simulation of the Kelvin-Helmholtz instability in finite-size objects: jets moving across the magnetic field in the middle of the field reversal layer; interaction between a magnetized plasma flow and a cylindrical plasma source with zero own magnetic field"#"hello world",
}

response = requests.post(url, data=form_data)
response.raise_for_status()

r = response.json()
print(clova_host.format(r["id"]))
response = requests.get(clova_host.format(r["id"]), stream=True)
response.raise_for_status()

with open("clova_test.mp3", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
