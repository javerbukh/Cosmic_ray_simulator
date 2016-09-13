from astropy.io import fits
from astropy.io.fits import update
import numpy as np
import argparse
import ConfigParser
import random

def get_config_file_names(config,args):
    """
    Reads in the config file specified in the command line arguement used to
    run this script. The default config file for this script is
    'set_cosmic_rays.cfg', so use that as a template for how to structure any
    custom config files
    """
    config.read(args.chosen_config)
    all_jumps = []
    all_multi_jumps = []
    input_file = ''
    output_file = ''
    for section in config.sections():
        if section == 'set_files':
            input_file = config.get('set_files', 'input_file')
            output_file = config.get('set_files', 'output_file')
            group_size = int(config.get('set_files', 'group_size'))
            source_slope = int(config.get(section, 'source_slope'))
            read_noise = float(config.get(section, 'read_noise'))
            expected_output_file = config.get('set_files', 'expected_output_file')
        elif section[0:2] == 'cr':
            pixel_x = int(config.get(section, 'pixel_x'))
            pixel_y = int(config.get(section, 'pixel_y'))
            group_val = int(config.get(section, 'group_val'))
            jump_size = float(config.get(section, 'jump_size'))

            curr_jump = pixel_x, pixel_y, group_val, jump_size
            all_jumps.append(curr_jump)
        elif section[0:3] == 'mcr':
            pixel_x1 = int(config.get(section, 'pixel_x1'))
            pixel_x2 = int(config.get(section, 'pixel_x2'))
            pixel_y1 = int(config.get(section, 'pixel_y1'))
            pixel_y2 = int(config.get(section, 'pixel_y2'))
            group_val = int(config.get(section, 'group_val'))
            jump_size = float(config.get(section, 'jump_size'))

            curr_jump = pixel_x1, pixel_x2, pixel_y1, pixel_y2, group_val, jump_size
            all_multi_jumps.append(curr_jump)

    return (input_file, output_file, expected_output_file, group_size, source_slope, read_noise, all_jumps, all_multi_jumps)

def apply_slope_and_noise(old_sci, num_groups, slope, noise):
    """
    Takes the slope and read noise from the config file and applies it
    to each pixel in the output file. The first group from the input file is
    used as the initial values that the rest of the groups are built off of
    """
    new_sci = np.zeros((len(old_sci), num_groups, 2048, 2048), dtype=np.float32)
    for integration in range(0, len(old_sci)):
        new_sci[integration,0] = old_sci[integration,0]
        electron_sum = np.zeros((2048,2048), dtype = np.float)
        for group in range(1,num_groups):
            electron_sum += np.random.poisson(slope, size = (2048,2048))
            gaussian_dist = np.random.normal(0.0, noise, (2048,2048))
            new_sci[integration,group] = new_sci[integration,0] + electron_sum + gaussian_dist
    return new_sci

def change_group_size(num_groups, source_slope, read_noise, input_file, output_file):
    """
    Used to increase or decrease the amount of groups per integration for
    testing purposes
    """
    hdulist = fits.open(input_file)

    curr_num_groups = hdulist['SCI'].data[0].shape[0]
    curr_num_integrations = hdulist['SCI'].data.shape[0]
    new_data = np.zeros((curr_num_integrations,num_groups,2048,2048), dtype=np.float32)

    new_data_for_groupdq = np.array(new_data, dtype=np.uint8)
    new_data_for_err = np.array(new_data, dtype=np.float32)
    hdulist['GROUPDQ'].data = new_data_for_groupdq
    hdulist['ERR'].data = new_data_for_err

    new_data = apply_slope_and_noise(hdulist['SCI'].data, num_groups, source_slope, read_noise)

    hdulist['SCI'].data = new_data
    hdulist.writeto(output_file, clobber=True)


def create_cosmic_rays(pixel_change, all_multi_jumps, output_file, expected_output_file, read_noise):
    """
    Adds Cosmic Rays (either single CRs or blocks of them) to the output_file and
    expected_output_file as specified in the Config file. Then, it updates the GROUPDQ
    header in expected_output_file for validation testing of the JWST pipeline later on
    """
    hdulist = fits.open(output_file)
    update_data = np.array(hdulist['SCI'].data)
    track_rays = np.zeros((len(update_data),len(update_data[0]),2048,2048), dtype=np.uint8)

    for (pixel_x, pixel_y, group_val, jump_size) in pixel_change:
        track_rays[0][group_val][pixel_x,pixel_y] = 4
        for val in range(group_val,len(update_data[0])):
            update_data[0][val][pixel_x,pixel_y] += jump_size

    for (pixel_x1, pixel_x2, pixel_y1, pixel_y2, group_val, jump_size) in all_multi_jumps:
        track_rays[0][group_val][pixel_x1:pixel_x2+1,pixel_y1:pixel_y2+1] = 4
        for val in range(group_val,len(update_data[0])):
            update_data[0][val][pixel_x1:pixel_x2+1,pixel_y1:pixel_y2+1] += jump_size


    hdulist['SCI'].data = update_data
    hdulist.writeto(output_file, clobber=True)
    hdulist['GROUPDQ'].data = track_rays
    hdulist.writeto(expected_output_file, clobber=True)
    return track_rays


################################################################################
# Main section
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("chosen_config")
args = parser.parse_args()

config = ConfigParser.ConfigParser()
(input_file, output_file, expected_output_file, group_size, source_slope, read_noise, all_jumps, all_multi_jumps) = get_config_file_names(config,args)

if group_size>2:
    change_group_size(group_size, source_slope, read_noise, input_file, output_file)

create_cosmic_rays(all_jumps, all_multi_jumps, output_file, expected_output_file, read_noise)

print ("Cosmic rays have been added to {}\n"
        "All changes were applied in the first integration".format(output_file))
