import os
import torch
import argparse

from utils.train import trainDexiNed
from utils.test import testDexiNed, testSegNet, evalConsistency, evalConsistencyFoldit
from utils.preprocess import extractImages
#import track


def parse_args():
	parser = argparse.ArgumentParser(description='Main pipline supports different function modes')
	parser.add_argument('--mode',
						type=str,
						default='train',
						help='[train] Train mode. [test] Test mode. [prep] Extract videos to frames. [eval] Evaluation. [track] Track frames (default: train)')

	parser.add_argument('--model',
						type=str,
						default='dexi',
						help='Specify model name to be trained or tested. (Options: dexi, foldit)')

	parser.add_argument('--closs',
						type=str,
						default='triplet',
						help='Set consistency loss function in training DexiNed (Options: triplet, quadruplet)')

	parser.add_argument('--ploss',
						type=str,
						default='bdcn',
						help='Set pixel-wise loss function in training DexiNed (Options: mse, bdcn)')

	parser.add_argument('--epsilon',
						type=float,
						default=0.00000001,
						help='Learning rate')

	parser.add_argument('--epochs',
						type=int,
						default=5,
						help='Learning rate')

	parser.add_argument('--weight',
						type=float,
						default=0.7,
						help='weight parameter in the loss function')

	parser.add_argument('--margin1',
						type=float,
						default=1.0,
						help='margin in triplet and margin 1 in quadruplet loss')

	parser.add_argument('--margin2',
						type=float,
						default=0.5,
						help='margin 2 in quadruplet loss')

	parser.add_argument('--img_width',
						type=int,
						default=256,
						help='Image width for training and testing.')
    
	parser.add_argument('--img_height',
						type=int,
						default=256,
						help='Image height for training and testing.')

	parser.add_argument('--mean_pixel_values',
						default=[103.939, 116.779, 123.68, 137.86],
						type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]

	parser.add_argument('--thr1',
						type=int,
						default=-1,
						help='threshold 1, default: -1 which means apply no threshold')

	parser.add_argument('--thr2',
						type=int,
						default=5,
						help='threshold 2, default: 5')

	parser.add_argument('--checkpoint',
						type=str,
						default='10_model.pth',
						help='Saved model path in checkpoint folder')

	parser.add_argument('--data_path',
						type=str,
						default=os.path.join('dataset_origin','train','TC'),
						help='Saved model path in checkpoint folder')

	parser.add_argument('--result_path',
						type=str,
						default=os.path.join('result','test'),
						help='Path to save result from DexiNed prediction')

	# FlowNet2 arguments
	parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
	parser.add_argument("--rgb_max", type=float, default=255.)

	args = parser.parse_args()
	
	return args


def main(args):
	# Setup device and cudnn
	if torch.cuda.device_count() == 0:
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	# Switch function
	if args.mode == 'train':
		if args.model == 'dexi':
			trainDexiNed(device, args)
		else:
			print('Undefined model name')

	elif args.mode == 'test':
		if args.model == 'dexi':
			testDexiNed(device, args)
		elif args.model == 'segnet':
			testSegNet(device, args)
		else:
			print('Undefined model name')

	elif args.mode == 'eval':
		if args.model == 'foldit':
			evalConsistencyFoldit(device, args)
		else:
			evalConsistency(device, args)

	elif args.mode == 'prep':
		basename = 'dataset_origin'
		paths = [os.path.join(basename,'train','TC'),
		         os.path.join(basename, 'val', 'TC'), 
		         os.path.join(basename,'test')]

		for path in paths:
			pathOut = path
			pathIn = os.path.join(pathOut, 'videos')
			extractImages(pathIn, pathOut)
	else:
		print('Undefined mode')
		#print("tracking mode to track edge features")
		#track.run(device, args)

if __name__ == '__main__':
	os.environ['KMP_DUPLICATE_LIB_OK']='True'
	args = parse_args()
	main(args)