import os
import numpy as np
import matplotlib.pyplot as plt


root = r".\lab_data"
omega_n = 20.
active_sensors = [1, 2, 3]


def analyse_documents(generic_file):
    for file_name in os.listdir(root):
        if file_name[:len(generic_file)] == generic_file and file_name[len(generic_file):-4].isdigit():
            metadata = np.genfromtxt(root + "\\" + file_name, delimiter=',', max_rows=2)
            data = np.genfromtxt(root + "\\" + file_name, delimiter=',', skip_header=2).transpose()

            spectra = np.fft.rfft(data)
            norm_spectra = np.abs(spectra * (spectra.shape[1] / np.sum(np.abs(spectra), axis=1).reshape((7,1))))
            phase = np.angle(spectra, deg=True) * (norm_spectra > 2.5)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            ax1.set_title(file_name.split(".")[0])

            ax1.set_xlabel("Frequency / Hz")
            ax2.set_xlabel(r"Relative frequency / $\omega$")

            ax1.legend(["Sensor " + str(i) for i in active_sensors])
            ax2.legend(["Sensor " + str(i) for i in active_sensors])

            max_phase_sample = int(2 * metadata[1] / omega_n)
            for i in active_sensors:
                ax1.plot(np.linspace(0., metadata[1] * 0.5, spectra.shape[1]), norm_spectra[i])
                ax2.plot(np.linspace(0., 2., max_phase_sample), phase[i, :max_phase_sample])
                # ax2.plot(np.linspace(0., metadata[1] * 0.5, spectra.shape[1]), phase[i])

            plt.show()


# low amplitude
generic_file = r"test_"

analyse_documents(generic_file)

# high amplitude
generic_file = r"test_la_"

# long shake
generic_file = r"test_long_p"
