import h5py
import numpy as np
from pathlib import Path
from time import perf_counter
import shutil
import sys
import argparse
import pandas as pd
from loguru import logger
import fnmatch




class BinnedMcStasData:
    def __init__(self, 
                 input_file: str, 
                 nbins:int = 50,
                 probability_weight:int = 10000,
                 chunkit: bool = None):
        self.input_file = Path(input_file)
        if not self.is_McStas_file(self.input_file):
            raise Exception("Something went wrong.")
        self.nbins = nbins
        self.npixels = 1280 * 1280
        self.probability_weight = probability_weight
        self.num_detectors = self.get_num_detectors()
        self.proton_charge = None
        self.max_t, self.min_t, self.max_p = (0.0,0.0,0.0)
        self.binned_data = np.zeros((3,self.npixels,self.nbins))
        self.chunkit = chunkit

    def is_McStas_file(self,file: Path) -> bool:
        """does the file exist? is it a McStas Nexus File?"""
        try:
            with h5py.File(file) as fp:
                dset = fp["entry1/data"]["Detector_0_event_signal_dat_list_p_x_y_n_id_t"]["events"]
        except OSError:
            logger.exception("Not an HDF5 file.")
            return False
        except KeyError:
            logger.exception("Data not found. Is it a McStas NeXuS file?")
            return False
        except FileNotFoundError:
            logger.exception("File not found")
            return False
        except Exception:
            logger.exception("Something went wrong.")
            return False
        return True

    def get_num_detectors(self):
        """
        the number of detectors is how many event lists with the following format:
        'Detector_?_event_signal_dat_list_p_x_y_n_id_t'
        """
        with h5py.File(self.input_file,'r') as fp:
            dgroup = fp["entry1/data"]
            dsets = list(dgroup.keys())
            filtered = fnmatch.filter(dsets, 'Detector_?_event_signal_dat_list_p_x_y_n_id_t')
            return len(filtered)

    def get_num_pixels(self):
        """ 
        this value is unlikely to change, but if needed, here's how to get it
        """
        with h5py.File(self.input_file,'r') as fp:
            dset = fp["entry1/instrument/components/nD_Mantid_0/output/BINS/pixels"][...]
            return len(dset.flatten())


    def min_max_probability_tof_chunked(self) -> tuple[float]:
        """
        Get the minumum and maxiumum tof and probability.
        Ignore zero tof and zero probabilities (likely errors in McStas).
        Done in chunks to avoid memory errors for large files.
        """
        max_p = 0.0
        max_t, min_t = 0.0, np.inf

        with h5py.File(self.input_file,"r") as fp:
            for i in range(self.num_detectors):
                logger.info(f"minmax_t frame {i}")
                dset = fp["entry1/data"][f"Detector_{i}_event_signal_dat_list_p_x_y_n_id_t"]["events"]

                for s in dset.iter_chunks():
                    chunk = dset[s]
                    chunk_t = chunk.T[5]
                    chunk_p = chunk.T[0]
                    tempmax_t = np.max(chunk_t)
                    tempmin_t = np.min(chunk_t[chunk_t > 0])
                    tempmax_p = np.max(chunk_p)
                    if tempmax_t > max_t:
                        max_t = tempmax_t
                    if tempmax_p > max_p:
                        max_p = tempmax_p
                    if tempmin_t < min_t:
                        min_t = tempmin_t
        logger.info(f"max_t: {max_t}, min_t: {min_t}")
        logger.info(f"max_p: {max_p}")
        return (max_t,min_t,max_p)
    
    def min_max_probability_tof(self) -> tuple[float]:
        """
        Get the minumum and maxiumum tof and probability.
        Ignore zero tof and zero probabilities (likely errors in McStas).
        """
        max_p = 0.0
        max_t, min_t = 0.0, np.inf

        with h5py.File(self.input_file,"r") as fp:
            for i in range(self.num_detectors):
                logger.info(f"minmax_t frame {i}")
                dset = fp["entry1/data"][f"Detector_{i}_event_signal_dat_list_p_x_y_n_id_t"]["events"][...]
                chunk_t = dset.T[5]
                chunk_p = dset.T[0]
                tempmax_t = np.max(chunk_t)
                tempmin_t = np.min(chunk_t[chunk_t > 0])
                tempmax_p = np.max(chunk_p)
                if tempmax_t > max_t:
                    max_t = tempmax_t
                if tempmax_p > max_p:
                    max_p = tempmax_p
                if tempmin_t < min_t:
                    min_t = tempmin_t
        logger.info(f"max_t: {max_t}, min_t: {min_t}")
        logger.info(f"max_p: {max_p}")
        return (max_t,min_t,max_p)
    
    def get_weight_parameter(self,max_p):
        """
        Get event weighting parameter for scaling event probabilities, normalized
        by setting the maximum probability to 1. probability_weight is an arbitrary
        value set to 1e5 as default.

        event_weights = probability_weight * (probabilities / max(probabilities))
        """

        return self.probability_weight / max_p
    
    def set_tof_bins(self):
        """
        set tof bins
        """
        return np.linspace(self.min_t,self.max_t,self.nbins)

    def do_tof_binning_chunked(self):
        """
        The actual tof-binning step.
        Done in chunks to avoid memory errors on large mcstas datasets.
        """
        ndet = self.num_detectors
        t1 = perf_counter()
        with h5py.File(self.input_file,"r") as fp:
            for j in range(ndet):
                tframe = perf_counter()
                logger.info(f"binning frame {j}")
                dset = fp["entry1/data"][f"Detector_{j}_event_signal_dat_list_p_x_y_n_id_t"]["events"]
                for s in dset.iter_chunks():
                    chunk = dset[s]
                    indexed = np.digitize(chunk.T[5],self.tof_bins)
                    for i,_ in enumerate(self.tof_bins):
                        f = chunk[np.where(indexed==i)]
                        # eventdata = np.bincount(f[:,4].astype(int),weights=f[:,0]*self.weight, minlength=self.npixels*ndet)
                        eventdata = np.bincount(f[:,4].astype(int),weights=f[:,0], minlength=self.npixels*ndet)
                        eventdata1 = eventdata[self.npixels*j:self.npixels*(j+1)]
                        self.binned_data[j,:,i] += eventdata1
                # for i in range(1,50):
                #     logger.debug(f"sum of frame{j}, tof {self.tof_bins[i-1]:.4f} - {self.tof_bins[i]:.4f}: {self.binned_data[j,:,i].sum()}")
                t2 = perf_counter()
                logger.info(f"time to process frame {j}: {t2-tframe:.2f} s")
        self.binned_data *= self.weight
        self.set_proton_charge()
        t3 = perf_counter()
        logger.info(f"time for all frames: {t3-t1:.2f} s")
    
    def do_tof_binning(self):
        """
        The actual tof-binning step.
        No chunking of data.
        """
        ndet = self.num_detectors
        t1 = perf_counter()
        with h5py.File(self.input_file,"r") as fp:
            for j in range(ndet):
                tframe = perf_counter()
                logger.info(f"binning frame {j}")
                dset = fp["entry1/data"][f"Detector_{j}_event_signal_dat_list_p_x_y_n_id_t"]["events"][...]
                indexed = np.digitize(dset.T[5],self.tof_bins)
                for i,_ in enumerate(self.tof_bins):
                    f = dset[np.where(indexed==i)]
                    eventdata = np.bincount(f[:,4].astype(int),weights=f[:,0], minlength=self.npixels*ndet)
                    eventdata1 = eventdata[self.npixels*j:self.npixels*(j+1)]
                    self.binned_data[j,:,i] = eventdata1
                # for i in range(1,50):
                #     logger.debug(f"sum of frame{j}, tof {self.tof_bins[i-1]:.4f} - {self.tof_bins[i]:.4f}: {self.binned_data[j,:,i].sum()}")
                t2 = perf_counter()
                logger.info(f"time to process frame {j}: {t2-tframe:.2f} s")
                dset = None
        self.binned_data *= self.weight
        self.set_proton_charge()
        t3 = perf_counter()
        logger.info(f"time for all frames: {t3-t1:.2f} s")

    def get_binned_data(self):
        return self.binned_data

    def set_proton_charge(self):
        self.proton_charge = (self.binned_data/self.probability_weight).sum(axis=(1,2))

    def get_proton_charge(self):
        return self.proton_charge
    
    def get_storage_memory_of_data(self):
        """
        get the storage size of the datasets to be loaded, in bytes.
        Will determine whether data to be chunked or not.
        """
        size = 0
        ndet = self.get_num_detectors()
        with h5py.File('/Users/aaronfinke/nmx_workflow/nmx_workflow/data/raw/main/mccode_phi0.h5') as fp:
            for i in range(ndet):
                dset = fp["entry1/data"][f"Detector_{i}_event_signal_dat_list_p_x_y_n_id_t"]["events"]
                size += dset.id.get_storage_size()
        return size
    
    def to_chunk(self,threshold = 20.0):
        """
        If the size of datasets is above a threshold (in GB),
        return True.
        """
        storage_size = self.get_storage_memory_of_data()
        storage_size = storage_size / (1<<30)   #convert to GB
        return storage_size > threshold

    
    def run(self):
        self.binned_data = np.zeros((self.num_detectors,self.npixels,self.nbins))
        logger.info(f"Storage size of datasets: {self.get_storage_memory_of_data()/(1<<30):.2f} GB")
        if self.chunkit or self.to_chunk(threshold=50):
            logger.info("Data will be processed in chunks.")
            self.max_t, self.min_t, self.max_p = self.min_max_probability_tof_chunked()
            self.weight = self.get_weight_parameter(max_p=self.max_p)
            self.tof_bins = self.set_tof_bins()
            self.do_tof_binning_chunked()
        else:
            self.max_t, self.min_t, self.max_p = self.min_max_probability_tof()
            self.weight = self.get_weight_parameter(max_p=self.max_p)
            self.tof_bins = self.set_tof_bins()
            self.do_tof_binning()


def parse_args():
    parser = argparse.ArgumentParser(description="Process McStas data with binning.")
    parser.add_argument("input_file", type=str, help="Path to the input McStas file.")
    parser.add_argument("--nbins", type=int, default=50, help="Number of bins for TOF binning.")
    parser.add_argument("--probability_weight", type=int, default=10000, help="Probability weight for event scaling.")
    return parser.parse_args()

def main():
    args = parse_args()
    binned_data = BinnedMcStasData(input_file=args.input_file,
                                    nbins=args.nbins, 
                                    probability_weight=args.probability_weight)
    binned_data.run()

if __name__ == "__main__":
    main()