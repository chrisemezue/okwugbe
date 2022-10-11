import os
import torch
import warnings
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from pathlib import Path
from sklearn.model_selection import train_test_split

class OkwugbeDataset(torch.utils.data.Dataset):
    """Create a Dataset for Okwugbe ASR.
    Args:
    data_type could be either 'test', 'train' or 'valid'
    """

    def __init__(self,transformation,
                target_sample_rate: int = None, 
                train_path: str =None,
                test_path: str = None,
                datatype: str = None,
                device: str = None, 
                validation_size: float =0.2):
        super(OkwugbeDataset, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.device = device
        self.validation_size = validation_size
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate
        self.original_data = None
        self.datatype = datatype.lower()

        if self.train_path==None:
            raise Exception(f"`train_path` cannot be empty! You provided no path to training dataset.")
        if self.datatype =='test':
            if self.test_path==None:
                warnings.warn(f"You provided no test set. test set will be set to None and not testing evaluation will be done.")
                self.test = None
            else:
                self.test = self.load_data(self.test_path,False)

        self.train, self.validation = self.load_data(self.train_path)
        if datatype.lower() == 'train':
            self.data = self.get_data(self.train, datatype)
            self.original_data = self.train

        if datatype.lower() == 'valid':
            self.data = self.get_data(self.validation, datatype)
            self.original_data = self.validation
            # Save the validation file
            VALID_FILE_NAME = Path(self.train_path).name
            self.save_pandas_to_csv(self.validation,f'/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/validation_data/VALID_{VALID_FILE_NAME}')  

        if datatype.lower() == 'test':
            if self.test is not None: 
                self.data = self.get_data(self.test, datatype)
                self.original_data = self.test
                # Save the test data
                VALID_FILE_NAME = Path(self.test_path).name
                self.save_pandas_to_csv(self.test,f'/home/mila/c/chris.emezue/okwugbe/hf_audio_classification/validation_data/TEST_{VALID_FILE_NAME}')  

            else:
                raise Exception(f"No test data was provided! Cannot request for test data")


        """datatype could be either 'test', 'train' or 'valid' """

    def load_data(self,path,split=True):
        training = pd.read_csv(path)

        if split:
            train,validation = train_test_split(training, test_size=self.validation_size,random_state=20)
            return train, validation
        else:
            return training

    def save_pandas_to_csv(self,df,filepath):
        df.to_csv(filepath,index=False)

    def get_data(self, dataset, datatype):
        data = dataset.to_numpy()
        print('{} set size: {}'.format(datatype.upper(), len(data)))
        return data

    def load_audio_item(self, d: list):
        utterance = int(d[1])
        wav_path = d[0]
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.to(self.device)

        waveform = self._resample_if_necessary(waveform, sample_rate)
        waveform = self.transformation(waveform,
                                    sampling_rate=self.transformation.sampling_rate, 
                                    max_length=16_000, 
                                    truncation=True,
                                    return_tensors="pt")
        return waveform.input_values, utterance

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, utterance)``
        """
        fileid = self.data[n]
        return self.load_audio_item(fileid)

    def __len__(self) -> int:
        return len(self.data)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    SAMPLE_RATE = 48000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = OkwugbeDataset(mel_spectrogram,
                            SAMPLE_RATE,
                            '/home/mila/c/chris.emezue/okwugbe/train_ibo.csv',
                            '/home/mila/c/chris.emezue/okwugbe/test_ibo.csv',
                            'train',
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]


