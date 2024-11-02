import os
import numpy as np


root = r".\lab_data"

def analyse_documents(generic_file):
    for file_name in os.listdir(root):
        if file_name[:len(generic_file)] == generic_file:
            data = np.genfromtxt(root + "\\" + file_name, delimiter=',', skip_header=2)

            frequency_list = 


# low amplitude
generic_file = r"test_"

analyse_documents(generic_file)

# high amplitude
generic_file = r"test_la_"

# long shake
generic_file = r"test_long_p"

import os
import numpy as np
import matplotlib.pyplot as plt


root = r".\lab_data"
omega_n = 20.
active_sensors = [1, 2]
normalize_to = 1

cutoff = 3.0
window_size = 21
granularity = 100
damping_range = 0.5

test_consistency = False
variable_omega_n = False


def slice_along_axis(ndim, axis, start, end):
    slices = [slice(None)] * ndim
    slices[axis] = slice(start, end)
    return tuple(slices)

def moving_average(a, n, axis=None):
    out = np.cumsum(a, axis=axis)
    if axis == None:
        start_slices = slice(n, None)
        end_slices = slice(None, -n)
        out_slices = slice(n-1, None)
    else:
        start_slices = slice_along_axis(a.ndim, axis, n, None)
        end_slices = slice_along_axis(a.ndim, axis, None, -n)
        out_slices = slice_along_axis(a.ndim, axis, n-1, None)
    out[start_slices] = out[start_slices] - out[end_slices]
    return out[out_slices] / n

def find_peaks(a, n, f=2.0, axis=-1):
    out = np.zeros_like(a)
    slices = slice_along_axis(a.ndim, axis, int(np.floor((n-1)*0.5)), -int(np.ceil((n-1)*0.5)))
    out[slices] = a[slices] > (f * moving_average(a, n, axis))
    return out

def first_peak(a, n, f=2.0, axis=-1):
    return np.argmax(find_peaks(a,n,f,axis))

def filtered(a):
    return a * (np.cumsum(a != 0, axis=1) == 1)

def analyse_documents(generic_file):
    spectra_sets = list()
    phase_sets = list()
    for file_name in os.listdir(root):
        if file_name[:len(generic_file)] == generic_file and file_name[len(generic_file):-4].isdigit():
            metadata = np.genfromtxt(root + "\\" + file_name, delimiter=',', max_rows=2)
            data = np.genfromtxt(root + "\\" + file_name, delimiter=',', skip_header=2).transpose()

            spectra = np.fft.rfft(data)
            abs_spectra = np.abs(spectra)
            norm_spectra = abs_spectra * (spectra.shape[1] / np.sum(abs_spectra, axis=1, keepdims=True))
            peaks = find_peaks(norm_spectra[normalize_to if normalize_to != None else slice(None)], window_size, cutoff)
            phase = np.angle(spectra, deg=True)

            if normalize_to == None:
                rel_spectra = norm_spectra * peaks
                rel_phase = phase * peaks
            else:
                rel_spectra = abs_spectra * peaks / abs_spectra[normalize_to]
                rel_phase = ((phase - phase[normalize_to]) % 360.) * peaks

            spectra_sets.append(rel_spectra[1:])
            phase_sets.append(rel_phase[1:])

            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            # ax2.set_title(file_name.split(".")[0])

            # ax1.set_xlabel("Frequency / Hz")
            # ax2.set_xlabel("Frequency / Hz")
            # ax3.set_xlabel(r"Relative Frequency / $\omega$")
            # ax4.set_xlabel(r"Relative Frequency / $\omega$")

            # max_phase_sample = int(omega_n * rel_spectra.shape[1] / (metadata[1] * np.pi))
            # for i in active_sensors:
            #     ax1.plot(np.linspace(0., metadata[1] * 0.5, spectra.shape[1]), norm_spectra[i])
            #     if i != normalize_to: ax2.plot(np.linspace(0., metadata[1] * 0.5, spectra.shape[1]), rel_spectra[i])
            #     ax3.plot(np.linspace(0., 2., max_phase_sample), phase[i, :max_phase_sample])
            #     if i != normalize_to: ax4.plot(np.linspace(0., 2., max_phase_sample), rel_phase[i, :max_phase_sample])

            # ax1.legend(["Sensor " + str(i) for i in active_sensors])
            # ax2.legend(["Sensor " + str(i) for i in active_sensors if i != normalize_to])
            # ax3.legend(["Sensor " + str(i) for i in active_sensors])
            # ax4.legend(["Sensor " + str(i) for i in active_sensors if i != normalize_to])

            # plt.show()

    spectra_sets = [filtered(s) for s in spectra_sets]
    phase_sets = [filtered(p) for p in phase_sets]

    acc, acc_contributors = np.zeros_like(spectra_sets[0]), np.zeros_like(spectra_sets[0])
    consistency = np.zeros((len(spectra_sets), *spectra_sets[0].shape))
    overlap = np.zeros((len(spectra_sets), spectra_sets[0].shape[0]))

    phase_acc, phase_acc_contributors = np.zeros_like(phase_sets[0]), np.zeros_like(phase_sets[0])
    phase_consistency = np.zeros((len(phase_sets), *phase_sets[0].shape))
    phase_overlap = np.zeros((len(phase_sets), phase_sets[0].shape[0]))

    for spectra in spectra_sets:
        acc += spectra
        acc_contributors += spectra != 0
    acc /= np.maximum(acc_contributors, 1)

    for phase in phase_sets:
        phase_acc += phase
        phase_acc_contributors += phase != 0
    phase_acc /= np.maximum(phase_acc_contributors, 1)

    if test_consistency:
        for i, spectra in enumerate(spectra_sets):
            contribution = spectra != 0
            consistency[i] = (acc_contributors > 1) * contribution * ((spectra / (acc + (acc == 0))) - 1.0)
            overlap[i] = np.sum(contribution, axis=1)
        total_deviation = np.sum(np.abs(consistency), axis=2) / overlap
        variance_deviation = np.var(consistency, axis=2, where=(consistency != 0))

        for i, phase in enumerate(phase_sets):
            phase_contribution = phase != 0
            phase_consistency[i] = (phase_acc_contributors > 1) * phase_contribution * (phase - phase_acc) / 180.
            overlap[i] = np.sum(phase_contribution, axis=1)
        phase_total_deviation = np.sum(np.abs(phase_consistency), axis=2) / overlap
        phase_variance_deviation = np.var(phase_consistency, axis=2, where=(phase_consistency != 0))

        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4)

        ax2.set_title("Total Consistency")
        ax3.set_title("Consistency Variation")
        ax5.set_title("Accumulated Response")
        ax6.set_title("Total Inconsistency")
        ax7.set_title("Consistency Variation")

        ax2.set_xlabel("Sensor Number")
        ax2.set_ylabel("Data Set")
        ax3.set_xlabel("Sensor Number")
        ax3.set_ylabel("Data Set")
        ax5.set_xlabel("Frequency / Hz")
        ax6.set_xlabel("Sensor Number")
        ax6.set_ylabel("Data Set")
        ax7.set_xlabel("Sensor Number")
        ax7.set_ylabel("Data Set")
    else:
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax2.set_title("Damping Prediction Alignment")

    ax1.set_title("Accumulated Response")

    ax1.set_xlabel("Frequency / Hz")

    wn_range = np.linspace(0.001, omega_n, granularity) if variable_omega_n else np.array([omega_n])
    z2s, ws, wns = np.meshgrid(np.linspace(0.001, damping_range, granularity)**2, np.linspace(0.0, np.pi * metadata[1], acc.shape[1]), wn_range)
    wr2s = (ws / wns)**2
    hs = np.sqrt((1 + 4*z2s*wr2s) / ((1-wr2s)**2 + 4*z2s*wr2s))
    mask = np.sum(acc != 0, axis=0) != 0
    hs /= np.tensordot(mask, hs, axes=1)
    sums = np.sum(acc, axis=1)

    min_index = np.min(np.argmax(acc > 0, axis=1))
    max_index = np.min(np.argmax(np.cumsum(acc, axis=1), axis=1))

    for i in active_sensors:
        if i != normalize_to:
            ax1.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1])[min_index:max_index], acc[i-1, min_index:max_index])

            if test_consistency:
                ax4.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1]), consistency[0][i - 1])
                ax5.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1])[min_index:max_index], phase_acc[i-1, min_index:max_index])
                ax8.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1]), phase_consistency[0][i - 1])
            else:
                alignment = np.sum((np.expand_dims(np.expand_dims(acc[i-1] / sums[i-1], axis=1), axis=2) - hs) ** 2, axis=0)
                mins = np.min(alignment, axis=0)
                selection_freq = np.argmin(mins)
                best_alignment = alignment[:, selection_freq]
                ax2.plot(np.linspace(0.001, damping_range, granularity), -np.power(best_alignment, 0.25))
                selection = np.argmin(best_alignment)
                print(f"Best frequency was {np.interp(selection_freq, [0, granularity-1], [0.001, omega_n]) if variable_omega_n else omega_n}")
                print(f"Best damping ratio was {np.interp(selection, [0, granularity-1], [0.0, damping_range])}")
                print(f"Lowest disagreement was {best_alignment[selection]}")
                ax1.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1])[min_index:max_index], (hs[:, :, selection_freq].T * (acc[i-1] != 0))[selection, min_index:max_index] * sums[i-1], linewidth=0.5)
                ax1.plot(np.linspace(0., metadata[1] * 0.5, acc.shape[1])[min_index:max_index], hs[:, :, selection_freq].T[selection, min_index:max_index] * sums[i-1], linewidth=0.5)

    if test_consistency:
        ax2.imshow(total_deviation)
        ax3.imshow(variance_deviation)
        ax5.legend(["Sensor " + str(i) for i in active_sensors if i != normalize_to])
        ax6.imshow(phase_total_deviation)
        ax7.imshow(phase_variance_deviation)

    ax1.legend(["Sensor " + str(i) for i in active_sensors if i != normalize_to])

    plt.show()


# low amplitude
generic_file = r"test_"
analyse_documents(generic_file)

# high amplitude
generic_file = r"test_la_"
analyse_documents(generic_file)

# long shake
generic_file = r"test_long_p"
