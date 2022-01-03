/*
 * This script applies a previously trained classifier
 * for HE images to all HE images that match a predefined
 * directory structure.
 * 
 * The structure is expected to look like this:
 * root	| Mouse 1	|
 * 			...
 * 		| Mouse N	|Histology	| Sample_1	|
 * 									...
 * 								| Sample_N	| HE filename
 * 	
 * 	The HE image is identified by the string "HE" in the filename
 * 	The output image is then saved into the newly created directory 
 * 	"..../Sample x/seg/"
 * 	
 * 	Steps that are taken:
 * 		- Image is openend in max resolution
 * 		- Image is saved as .hdf5 format for Ilastik to understand
 * 		- Ilastik is called to segment image
 * 		- hdf5 image is deleted again to save storage
 * 		- Output image is moved to output directory
 */

// Config
close("*");
roiManager("reset");

var usebatch =  false;
var hasbeencropped = false;
setBatchMode(usebatch);

type = "IF";	// which type of image is segmented? "HE" or "IF"
reseg = true; 	// if true, segmentation will be done all over again. If false, only unprocessed samples will be segmented

// Ilastik files
root = "E:/Promotion/Projects/2020_Radiomics/Data/";
classifier = root + "../Scripts/Histology/Classify/"+type+"_classifier_AD_JM_2.ilp";
Ilastik_exe = "C:/Program Files/ilastik-1.3.3post3/run-ilastik.bat";

// Input dir
mice = getsubdirs(root);

// iterate over all mice
for (i = 0; i < mice.length; i++) {

	print(d2s(i+1, 0) + "/" + d2s(mice.length, 0) + " animals");
	dir_sample = root + mice[i] + "Histology/";
	samples = getsubdirs(dir_sample);

	// iterate over samples (aka tissue sections
	for (j = 0; j < samples.length; j++) {
		fnames = findFileByID(dir_sample + samples[j], type);

		// Check if all images are provided:
		// IF HE or IF is missing, there's no need for segmentation
		files = getFileList(dir_sample + samples[j]);
		if (files.length <2) {
			print("    Probably some data missing in " + dir_sample + samples[j]);
			continue;
		}

		// check plausibility of filenames
		if (fnames.length>1) {
			intrpt = getBoolean("    Something's weird here. Check?");
			if (intrpt) {
				exit();
			} else {
				continue;
			}
		}

		if (fnames.length == 0) {
			print("    ---> No IF image found here");
			continue;
		}
		
		fname = fnames[0];  // extract fname from single-element array <fnames>

		// Make output directory ( if it doesn't already exist)
		seg_dir = dir_sample + samples[j] + "1_seg/";
		if (!File.exists(seg_dir)) {
			File.makeDirectory(seg_dir);
		}

		// Check if seg has been done before
		if (!(reseg || ! File.exists(seg_dir + type + "_seg_Simple_Segmentation.tif"))) {
			print("    Has been segmented before. Skipping.");
			continue;
		}

		// (Pre-) process
		image = Load(dir_sample + samples[j] + fname);	// Load image
		
		// Normalize if IF and identify black tiles if HE
		if (type == "IF") {	image = Normalize(image); }
		if (type == "HE") {	getTileBackground(image, seg_dir); }

		
		// If a ROI has been provided
		if ((roiManager("count") > 0) && (!hasbeencropped)) {
			roiManager("save selected", dir_sample + samples[j] + File.getNameWithoutExtension(fname) + ".roi");
			roiManager("select", 0);
			run("Crop");
		}		
		
		image = getTitle();
		f_hdf5 = ExportH5(image, seg_dir + type + "_seg");	// convert to H5 and save
		RunIlastik(Ilastik_exe, classifier, f_hdf5);  //  Call Ilastik

		// Clean up
		close("*");
		roiManager("reset");
	}
}

function RunIlastik(IL_executable, classifier, filename){
	// Runs Ilastik <executable> to apply <classifier> on <filename>
	print("    Running pixel classification on " + filename);
	outpath = File.getDirectory(filename);
	exec(IL_executable, 
			"--headless",
			"--project=" + classifier,
			filename, 
			"--output_format=tif",
			"--export_source=Simple Segmentation",
			"--output_filename_format="+outpath+"/{nickname}_Simple_Segmentation");	
	File.delete(filename);	// remove h5 image to save storage
}

function getTileBackground(image, dst_dir){
	selectWindow(image);
	setSlice(1);
	run("Duplicate...", "duplicate title=TileBackground0 channels=1");
	run("Duplicate...", "duplicate title=TileBackground1 channels=1");

	// identify pixels with value 0
	selectWindow("TileBackground0");
	setThreshold(0, 0);
	run("Convert to Mask");

	// identify pixels with value 255
	selectWindow("TileBackground1");
	setThreshold(255, 255);
	run("Convert to Mask");

	imageCalculator("OR create", "TileBackground0","TileBackground1");
	result = getTitle();
	close("TileBackground0");
	close("TileBackground1");

	// extract ROI
	selectWindow(result);
	run("Erode");
	setThreshold(128, 255);
	run("Create Selection");
	
	// smooth ROI outline
	run("Enlarge...", "enlarge=-30 pixel");
	run("Enlarge...", "enlarge=60 pixel");  // expand ROI by 20 pixels to include edges in segmented image
	roiManager("add");
	close(result);

	roiManager("save", dst_dir + "BlackTiles.zip");
	
}

function ExportH5(img, f_output){
	// saves input image <image> as <f_output>.h5
	selectWindow(img);
	run("Export HDF5", 	"select="+f_output+".h5 " + 
						"exportpath=" + f_output + ".h5 " + 
						"datasetname=data " +
						"compressionlevel=0 " +
						"input=["+img+"]");
	return f_output +".h5";
}

function Normalize (image){
	/*
	 * Function for the preprocessing of IF image data.
	 * Images are re-sorted into RGB stacks, converted to 16-bit
	 * and normalized. Brightest pixels are clipped to eliminate
	 * the influence of staining artifacts.
	 */

	selectWindow(image);
	if (nSlices ==1) {
		run("RGB Stack");
	}
	run("Make Composite");
	run("16-bit");
	
	// do BG subtraction and normalization
	// check if sample comes from Slidescanner
	N = nSlices;
	if (N == 4){
		C0 = 2;
		C1 = 4;
	} else {
		C0 = 1;
		C1 = 3;
	}

	// in every loop, copy channel, subtract background and normalize
	index = 1;
	for (i = C0; i <= C1; i++) {
		selectWindow(image);
		setSlice(i);
		run("Duplicate...", "title=" + index);
		selectWindow(index);
		run("Subtract Background...", "rolling=50 stack");
		run("Enhance Contrast...", "saturated=0.3 normalize");

		index += 1;
	}
	exit();

	// if it's slidescanner data (which is typically badly cropped)
	// give user chance to draw his own ROI
	if (N == 4) {
		setBatchMode(false);
		selectWindow("1");
		selectWindow("2");
		selectWindow("3");
		waitForUser("Set a ROI if image is badly cropped.\n"+
					"Or do nothing if it's fine");

		if (selectionType() != -1) {
			roiManager("add");
			roiManager("select", 0);
		}
	}
	setBatchMode(usebatch);
	
	close(image);
	run("Merge Channels...", "c1=1 c2=2 c3=3 create");
	rename(image);
	return image;
}

function Load(f_input){
	// converts an input file <f_input> into an hdf5 file <f_ouput>.
	// The ".h5" file extension will be added on the fly.
	//print("    --->"+f_input);

	hasbeencropped = false;
	
	run("Bio-Formats Importer", "open="+f_input+" autoscale "+
				"color_mode=Default "+
				"rois_import=[ROI manager] "+
				"view=Hyperstack "+
				"stack_order=XYCZT "+
				"series_1");
	
	dir = File.getDirectory(f_input);
	f  = File.getNameWithoutExtension(f_input);

	if (File.exists(dir + f + ".roi")) {
		roiManager("open", dir + f + ".roi");
		roiManager("select", 0);
		run("Crop");
		hasbeencropped = true;
	}
	
	return getTitle();
}

function findFileByID(path, ID){
	// browse directory <path> and check for files that match the given ID string <ID>
	files = newArray();
	filelist = getFileList(path);
	for (i = 0; i < filelist.length; i++) {

		// exclude ROIs from search
		if (endsWith(filelist[i], "roi")) {
			continue;
		}
		
		if (matches(filelist[i], ".*" + ID + ".*")) {
			files = Array.concat(files, filelist[i]);
		}
	}

	print("    Found " + d2s(files.length, 0) +" files matching " + ID + " in " + path);
	Array.show(files);
	return files;
}

function getsubdirs(path){
 	// Returns array with all subdirectories in <path>
 	list = getFileList(path);
 	dirlist = newArray();
 	for (i = 0; i < list.length; i++) {
 		if (File.isDirectory(path + list[i])) {
 			dirlist = Array.concat(dirlist, list[i]);
 		}
 	}
 	return dirlist;
 }
