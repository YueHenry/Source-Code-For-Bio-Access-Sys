'''
This part is used to train the speaker model and evaluate the performances
'''
import itertools
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn

import tools
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from model import *
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter()
from dataLoader import *
import scipy.io.wavfile as wav

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
			tags = ["train_loss", "train_acc", "learning_rate"]
			tb_writer.add_scalar(tags[0], loss/(num), epoch)
			tb_writer.add_scalar(tags[1], top1/index*len(labels), epoch)
			tb_writer.add_scalar(tags[2], lr, epoch)
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)


	def eval_network(self, eval_list, eval_path):
		self.eval()
		self.to("cpu")
		files = []
		embeddings = {}
		feats = {}
		lines = open(eval_list).read().splitlines()
		# print("lines",lines)
		for line in lines:
			# print("line", line)
			files.append(line.split()[1])
			# print("files1",files)
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()
		# print("setfiles", setfiles)

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			# print("idx", idx)
			# print("file",file)
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0))#.cuda()##Torchaudio fbank
			# data_1 = torch.FloatTensor(audio).cuda()##pncc
			# print("data_1",data_1.shape)
			# data_1=data_1.resize()
			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				# print("asf", asf)
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats)#.cuda()
			# print("data_1",data_1.shape)
			# print("data_2", data_2.shape)
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				# print("embedding_1.shape",embedding_1.shape)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
				# print("embedding_2.shape", embedding_2.shape)
			embeddings[file] = [embedding_1, embedding_2]
			# print("embeddings[file].shape",len(embeddings[file]))
		scores, labels  = [], []
		positive_scores=[]
		negative_scores=[]

		for line in lines:
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			# print("line.split()[1]",line.split()[1])
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# print("line.split()[2]",line.split()[2])
			# print("embedding_11",embedding_11.shape)
			# print("embedding_12", embedding_12.shape)
			# print("embedding_21", embedding_21.shape)
			# print("embedding_22", embedding_22.shape)
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			if int(line.split()[0])==1:
				positive_scores.append(score)
			else:
				negative_scores.append(score)
		# print(sum(positive_scores)/len(positive_scores))
		# print(sum(negative_scores)/len(negative_scores))
		print(max(positive_scores))
		print(min(positive_scores))

		return max(positive_scores)
		# print(max(negative_scores))
		# print(min(negative_scores))
		# print("scores",scores)
		# print("labels", labels)
		# np.savetxt('./exps/exp18/VoxMovies-40/labels-e5.txt',labels)
		# np.savetxt('./exps/exp18/VoxMovies-40/scores-e5.txt', scores)
		# tunedThreshold, EER, fpr, fnr = tuneThresholdfromScore(scores, labels, [1, 0.1])
		#
		# print(tunedThreshold)
		# fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		# minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.1, 1,1)
		# return EER, minDCF #fpr,tpr,

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)