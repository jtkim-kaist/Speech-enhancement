import matlab.engine
import librosa
# import matlab.engine as mateng


def se_eval(clean, noisy, fs):

    eng = matlab.engine.start_matlab()
    clean = matlab.double(clean.tolist())
    noisy = matlab.double(noisy.tolist())
    pesq, stoi, ssnr, lsd = eng.se_eval(clean, noisy, fs, nargout=4)

    measure = {"pesq": pesq, "stoi": stoi, "ssnr": ssnr, "lsd": lsd}

    return measure


fs = 16000.0

clean, _ = librosa.load('./clean.wav', fs)
noisy, _ = librosa.load('./noisy.wav', fs)

measure = se_eval(clean, noisy, fs)
