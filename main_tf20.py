import tensorflow as tf
import os
import pathlib
import time
import datetime
import sys
import logging
import datetime
from glob import glob
import numpy as np
import argparse

#origin libs
from models.model256 import MODEL256 as MODEL
#from models.model512 import MODEL512 as MODEL
from common.utils import *

#------------PARM SETTING--------------
#log-level(INFO,WARNING,ERROR,DEBUG,CRITICAL)
log_level = logging.INFO

#log directory
log_dir = 'logs'
#checkpoint directory
checkpoint_dir = './training_checkpoints'
#input directory of learning picture for training
train_input_dir = 'train_input_tocolor'
#input directory of learning picture for testing
test_input_dir = 'test_input_tocolor'
#input file name pattern
input_fname_pattern = '*.png'
#output directory of generated picture
output_dir = 'out'

#save checkpoint interval(unit:epoch)
ckpt_num = 100
#checkpoint file max max_to_keep
max_to_keep = 2
#generate picture interval(unit:epoch)
save_pic_num = 100

#number of times epoch
epochs = 20000
#batch size to learn picture
BATCH_SIZE = 64

#hyper parameters
gen_lr = 2e-4
gen_beta1 = 0.5
disc_lr = 2e-4
disc_beta1 = 0.5
LAMBDA = 100

#for google colab prefix
#add_dir_prefix='pix2pix_tf20/'
add_dir_prefix=''

#--------------------------
parser = argparse.ArgumentParser(description='PIX2PIX')
parser.add_argument('--runmode', required=True, help='rum mode [first, again, generate], first=at the first time learning. again=start from checkpoint. generate=genrate picture from checkpoint', choices=['first','again','generate'])
parser.add_argument('--log_dir', help='log directory')
parser.add_argument('--ckpt_dir', help='checkpoint directory')
parser.add_argument('--train_input_dir', help='input directory of learning picture for training')
parser.add_argument('--test_input_dir', help='input directory of learning picture for testing')
parser.add_argument('--output_dir', help='output directory of generated picture')
parser.add_argument('--save_ckpt_num', type=int, help='save checkpoint interval(unit:epoch)')
parser.add_argument('--ckpt_keep_num', type=int, help='checkpoint file max max_to_keep')
parser.add_argument('--save_pic_num', type=int, help='generate picture interval(unit:epoch)')
parser.add_argument('--epochs_num', type=int,help='number of times epoch')
parser.add_argument('--batch_size', type=int, help='batch size to learn picture')

args = parser.parse_args()

if args.log_dir is not None:
    log_dir = args.log_dir
if args.ckpt_dir is not None:
    checkpoint_dir = args.ckpt_dir
if args.train_input_dir is not None:
    train_input_dir = args.train_input_dir
if args.test_input_dir is not None:
    test_input_dir = args.test_input_dir
if args.output_dir is not None:
    output_dir = args.output_dir
if args.save_ckpt_num is not None:
    ckpt_num = args.save_ckpt_num
if args.ckpt_keep_num is not None:
    max_to_keep = args.ckpt_keep_num
if args.save_pic_num is not None:
    save_pic_num = args.save_pic_num
if args.epochs_num is not None:
    epochs = args.epochs_num
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size

ERR_FLG = False
log_dir = add_dir_prefix+log_dir
if os.path.isdir(log_dir) == False:
    os.makedirs(os.path.join(log_dir))
log_prefix = os.path.join(log_dir, "system-{}.log".format(timestamp()))
logging.basicConfig(filename=log_prefix, level=log_level)

_train_data_path = os.path.join(add_dir_prefix+train_input_dir)
train_data_path = os.path.join(add_dir_prefix+train_input_dir, input_fname_pattern)
if os.path.isdir(_train_data_path) == False:
    print("ERROR:TRAIN DIRECTORY is not found : {}".format(_train_data_path))
    ERR_FLG = True
train_data = glob(train_data_path)
if len(train_data) == 0:
    print("ERROR:[!] No data found in '" + train_data_path + "'")
    ERR_FLG = True

_test_data_path = os.path.join(add_dir_prefix+test_input_dir)
test_data_path = os.path.join(add_dir_prefix+test_input_dir, input_fname_pattern)
if os.path.isdir(_test_data_path) == False:
    print("ERROR:TEST DIRECTORY is not found : {}".format(_test_data_path))
    ERR_FLG = True
test_data = glob(test_data_path)
if len(test_data) == 0:
    print("ERROR:[!] No data found in '" + test_data_path + "'")
    ERR_FLG = True

checkpoint_prefix = os.path.join(add_dir_prefix+checkpoint_dir)
if os.path.isdir(checkpoint_prefix) == False:
    os.makedirs(checkpoint_prefix)

data_path = os.path.join(add_dir_prefix+output_dir+'/img')
if os.path.isdir(data_path) == False:
    os.makedirs(data_path)

if ERR_FLG == True:
    print("please fix error. [program exit]")
    #sys.stdout.write(str(1))
    sys.exit(1)

np.random.shuffle(train_data)
TRAIN_BUFFER_SIZE = len(train_data)
np.random.shuffle(test_data)
TEST_BUFFER_SIZE = len(test_data)

model = MODEL()

generator, discriminator = model.gen_gene_and_deisc()
generator.summary(print_fn=lambda x: logging.info('{}'.format(x)))
generator.summary()
discriminator.summary(print_fn=lambda x: logging.info('{}'.format(x)))
discriminator.summary()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=gen_beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr, beta_1=disc_beta1)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=max_to_keep)

def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    _timestamp = timestamp()
    save_images(prediction, (1,1),
        '{}/train_{:08d}_{}.png'.format(data_path, epoch, _timestamp))
    output = "./img/train_{:08d}_{}.png".format(epoch, _timestamp)
    save_images(test_input, (1,1),
        '{}/train_{:08d}_{}-input.png'.format(data_path, epoch, _timestamp))
    input = "./img/train_{:08d}_{}-input.png".format(epoch, _timestamp)

    return input, output

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def fit(train_ds, test_ds):
    for epoch in range(epochs):
        start = time.time()
        cnt = 0

        if TRAIN_BUFFER_SIZE < BATCH_SIZE:
            logging.error ("[!] Entire dataset size is less than the configured batch_size")
            raise Exception("[!] Entire dataset size is less than the configured batch_size")
        batch_idxs = TRAIN_BUFFER_SIZE // BATCH_SIZE
        for idx in range (int(batch_idxs)):
            batch_start = time.time()
            sample_files = train_ds[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            #dataset = tf.data.Dataset.from_tensor_slices(sample_files).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
            #train_dataset = tf.data.Dataset.list_files(train_prefix+'/*.png')
            train_dataset = tf.data.Dataset.list_files(sample_files)
            ##dataset: <_ShuffleDataset element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
            ##dataset: <_ShuffleDataset element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
            train_dataset = train_dataset.map(load_image_train,
                                              num_parallel_calls=tf.data.AUTOTUNE)
            train_dataset = train_dataset.shuffle(BATCH_SIZE)
            train_dataset = train_dataset.batch(BATCH_SIZE)

            for step, (input_image, target) in train_dataset.take(BATCH_SIZE).enumerate():
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, step)

            #print ('Batch Step(size:{}) :{}/{} in epoch {} is {} sec'.format(BATCH_SIZE, idx+1, batch_idxs, epoch+1, time.time()-batch_start))
            logging.info ('Batch Step(size:{}) :{}/{} in epoch {} is {} sec'.format(BATCH_SIZE, idx+1, batch_idxs, epoch+1, time.time()-batch_start))

        logging.info('gen_total_loss:{}'.format(gen_total_loss))
        logging.info('gen_gan_loss:{}'.format(gen_gan_loss))
        logging.info('gen_l1_loss:{}'.format(gen_l1_loss))
        logging.info('disc_loss:{}'.format(disc_loss))
        #print('gen_total_loss:', gen_total_loss)
        #print('gen_gan_loss:', gen_gan_loss)
        #print('gen_l1_loss:', gen_l1_loss)
        #print('disc_loss:', disc_loss)

        if (epoch+1) % save_pic_num == 0:
            np.random.shuffle(test_ds)
            test_ds_tmp = tf.data.Dataset.list_files(test_ds[0])
            test_ds_tmp = test_ds_tmp.map(load_image_test)
            test_ds_tmp = test_ds_tmp.batch(1)
            example_input, example_target = next(iter(test_ds_tmp.take(1)))
            generate_images(generator, example_input, example_target, epoch)
            logging.info("image saved!")
            #print("image saved!")
        # Save (checkpoint) the model
        if (epoch + 1) % ckpt_num == 0:
            logging.info('save checkpoint:{}'.format(checkpoint_prefix))
            #print('save checkpoint:{}'.format(checkpoint_prefix))
            manager.save()

        #print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def load_c(checkpoint_dir):
    logging.info(" [*] Reading checkpoints...{}".format(checkpoint_dir))
    print(" [*] Reading checkpoints...{}".format(checkpoint_dir))
    checkpoint_prefix_load = os.path.join(add_dir_prefix+checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_prefix_load)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix_load))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        counter = int(ckpt_name.split('-')[-1])
        logging.info("******** [*] Success to read {}".format(ckpt_name))
        print("******** [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        logging.error(" [*] Failed to find a checkpoint")
        print(" [*] Failed to find a checkpoint")
        return False, 0

def append_index(input, output):
    index_path = os.path.join(add_dir_prefix+output_dir, "index.html")

    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>input</th><th>output</th></tr>")

    index.write("<td><img src='{}'></td>".format(input))
    index.write("<td><img src='{}'></td>".format(output))
    index.write("</tr>")
    return index_path

def main(args):
    if args.runmode == 'again' or args.runmode == 'first' or args.runmode == 'generate':
        if args.runmode == 'again':
            flag, counter = load_c(checkpoint_dir)
            if flag:
                logging.info("# re-learning start")
                print("# re-learning start")
                try:
                    fit(train_data, test_data)
                except BaseException as e:
                    print(e)
                    logging.error(e, stack_info=True)
            else:
                logging.error("stop. reason:failed to load")
                print("stop. reason:failed to load")
        elif args.runmode == 'first':
            logging.info("# first learning start")
            print("# first learning start")
            try:
                fit(train_data, test_data)
            except BaseException as e:
                print(e)
                logging.error(e, stack_info=True)
        elif args.runmode == 'generate':
            flag, counter = load_c(checkpoint_dir)
            if flag:
                logging.info("# re-learning start")
                print("# image-generate start")
                try:
                    SMPL_SIZE=1
                    for idx in range (int(len(test_data))):
                        test_ds_tmp = test_data[idx*SMPL_SIZE:(idx+1)*SMPL_SIZE]
                        #dataset = tf.data.Dataset.from_tensor_slices(sample_files).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
                        #train_dataset = tf.data.Dataset.list_files(train_prefix+'/*.png')
                        test_ds_tmp = tf.data.Dataset.list_files(test_ds_tmp)
                        test_ds_tmp = test_ds_tmp.map(load_image_test)
                        test_ds_tmp = test_ds_tmp.batch(1)
                        example_input, example_target = next(iter(test_ds_tmp.take(1)))
                        input, output = generate_images(generator, example_input, example_target, idx)
                        append_index(input, output)
                except BaseException as e:
                    print(e)
                    logging.error(e, stack_info=True)

if __name__ == '__main__':
    #args = sys.argv
    #args.runmode='first'
    main(args)
