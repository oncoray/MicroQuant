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
 */

// Config
close("*");
roiManager("reset");

var usebatch =  false;
var downsamplingfactor = 4;
var registration_width = 5000;
var width;
var height;
var rereg = true; 	// do registration all over again?

setBatchMode(usebatch);
run("CLIJ2 Macro Extensions", "cl_device=");

var JCs = newArray();

// Elastix files
root = "E:/Promotion/Projects/2020_Radiomics/Data/";
elastix_params = "E:/Promotion/Projects/2020_Radiomics/Scripts/Histology/Register/elastix_parameters.txt";
elastix_exe = "E:/Promotion/Projects/2020_Radiomics/Scripts/Histology/Register/elastix-5.0.1-win64/elastix.exe";
transformix_exe = "E:/Promotion/Projects/2020_Radiomics/Scripts/Histology/Register/elastix-5.0.1-win64/transformix.exe";

// Input dir
mice = getsubdirs(root);

// iterate over all mice
for (i = 8; i < mice.length; i++) {

	print(d2s(i+1, 0) + "/" + d2s(mice.length, 0) + " animals");
	dir_sample = root + mice[i] + "Histology/";
	samples = getsubdirs(dir_sample);

	// iterate over samples (aka tissue sections)
	for (j = 0; j < samples.length; j++) {
		
		// check if segmentations exist
		if (!File.exists(dir_sample + samples[j] + "1_seg/")) {
			print("    " + dir_sample + samples[j] +  "wasn't segmented.");
			continue;
		}

		if (matches(samples[j], ".*censored.*")) {
			print("    " + dir_sample + samples[j] +  "was censored.");
			continue;
		}

		// define directories
		input_dir = dir_sample + samples[j] + "1_seg/";
		output_dir = dir_sample + samples[j] + "2_reg/";
		
		fmoving = findFileByID(input_dir, "IF_seg_Simple");
		ftarget = findFileByID(input_dir, "HE_seg_Simple");

		// Check if all images are provided: IF HE or IF is missing, there's no need for registration
		if ((fmoving.length == 0) || (ftarget.length == 0)) {
			print("    " + "Missing image data in " + input_dir);
			continue;
		}

		fmoving = fmoving[0];
		ftarget = ftarget[0];

		/*
		if (!matches(input_dir, ".*N182b_SAS_08.*")) {
			continue;
		}
		*/

		// Make output directory ( if it doesn't already exist)
		if (!File.exists(output_dir)) {
			File.makeDirectory(output_dir);
		}

		
		// Check if reg has been done before
		if (!(rereg || ! File.exists(output_dir + "IF_seg_Simple_Segmentation_transformed.tif"))) {
			print("    Has been registered before. Skipping.");
			continue;
		}
		
		// Start masking process, memorize image dimensions
		roiManager("open", input_dir + "BlackTiles.zip");	// get black tile segmentation
		print(input_dir + fmoving);
		print(input_dir + ftarget);
		moving = Load(input_dir + fmoving);
		target = Load(input_dir + ftarget);
		getDimensions(width, height, channels, slices, frames);

		DSF_target = Mask(target, 3, true);
		DSF_moving = Mask(moving, 1, false);
		downsamplingfactor = DSF_target;

		img_reg = register(moving, target, input_dir + fmoving, output_dir);
		//postproc_IF(img_reg);

		// Clean up
		close("*");
		roiManager("reset");
		Ext.CLIJ2_clear();
	}
}

function postproc_IF(fname){
	// postporcessing for registered and upsampled IF
	// segmentation images. Images are stored with LZW compression
	// so this should free quite a bit of storage

	open(fname);
	source = getTitle();
	Ext.CLIJ2_clear();
	Ext.CLIJ2_push(source);

	// convert from signed to unsigned uint16, set minium to zero
	Ext.CLIJ2_convertUInt16(source, uint16);	
	Ext.CLIJ2_getMinimumOfAllPixels(uint16, min);
	Ext.CLIJ2_addImageAndScalar(uint16, tmp, (-1)*min);
	Ext.CLIJ2_convertUInt8(tmp, uint8);
	close(source);
	Ext.CLIJ2_pull(uint8);
	source = getTitle();
	
	mask_CD31 = CLIJ2_threshold_range(source, 3);
	mask_Hoe = CLIJ2_threshold_range(source, 2);
	mask_Pimo = CLIJ2_threshold_range(source, 4);

	// Additional blurring for Hoechst channel
	
	Ext.CLIJ2_clear();
	Ext.CLIJ2_push(mask_Hoe);
	Ext.CLIJ2_multiplyImageAndScalar(mask_Hoe, tmp, 255);  // rescale to 0-255
	Ext.CLIJ2_gaussianBlur2D(tmp, mask_Hoe, 75, 75);
	Ext.CLIJ2_threshold(mask_Hoe, tmp, 0.1*255);

	close(mask_Hoe);
	Ext.CLIJ2_pull(tmp);
	mask_Hoe = getTitle();
	Ext.CLIJ2_clear();

	run("Merge Channels...", "c1="+mask_CD31+" c2="+mask_Pimo+" c3="+mask_Hoe+" create");
	if (File.exists(fname)) {
		File.delete(fname);	
	}
	run("Bio-Formats Exporter", "save="+fname+" compression=LZW");

}

function CLIJ2_threshold_range(source, val){
	//selectWindow(image);
	
	Ext.CLIJ2_clear();
	Ext.CLIJ2_push(source);
	Ext.CLIJ2_labelToMask(source, tmp, val);
	Ext.CLIJ2_openingDiamond(tmp, output, 2);
	Ext.CLIJ2_pull(output);
	
	return getTitle();
}



function register(moving_img, target_img, moving_data, outdir) {
	print("    ---> Registration and Transforming.");
	selectWindow(moving_img);
	saveAs("tif", outdir + moving_img);
	close();

	selectWindow(target_img);
	saveAs("tif", outdir + target_img);
	close();

	exec(elastix_exe + " " +  
		"-p " + elastix_params + " " +
		"-out "+ outdir + " " +
		"-m " + outdir + moving_img + ".tif" + " " +
		"-f " + outdir + target_img + ".tif");

	JC = calc_JC(outdir + target_img + ".tif", outdir + "result.0.tif");
	JCs= Array.concat(JCs,JC);
	Array.show(JCs);


	// Now, manipulate transform file so that it fits the large images
	trafofile = outdir + "TransformParameters.0.txt";
	f = File.openAsRawString(trafofile);
	lines = split(f, "\n");
	File.delete(trafofile);
	
	newtrafo = File.open(trafofile);
	
	for (i = 0; i < lines.length; i++) {
		line = lines[i];

		// change trafo params
		if (startsWith(line, "(TransformParameters")) {
			line = substring(line, 1, line.length-1);
			line = split(line, " ");
			print(newtrafo, "(TransformParameters " + 
					line[1] + " " + line[2] + " " + 
					line[3] + " " + line[4] + " " + 
					downsamplingfactor*parseFloat(line[5]) + " " + 
					downsamplingfactor*parseFloat(line[6]) + ")");
			continue;  
		}

		// change image sizes
		if (startsWith(line, "(Size")) {
			print(newtrafo, "(Size " + width + " " + height +")");
			continue;
		}

		// change rotation center
		if (startsWith(line, "(CenterOfRotationPoint")) {
			line = substring(line, 1, line.length-1);
			line = split(line, " ");
			print(newtrafo, "(CenterOfRotationPoint " + 
					downsamplingfactor*parseFloat(line[1]) + " " + 
					downsamplingfactor*parseFloat(line[2]) + ")");
			continue;
		}

		// change interpolation
		if (startsWith(line, "(ResampleInterpolator")) {
			// print(newtrafo, line);
			print(newtrafo, "(FinalLinearInterpolator NearestNeighborInterpolator)");
			continue;
		}

		// change result pixel format
		if (startsWith(line, "(ResultImagePixelType")) {
			// print(newtrafo, line);
			print(newtrafo, "(ResultImagePixelType short)");
			continue;
		}

		// change compression setting
		if (startsWith(line, "(CompressResultImage")) {
			// print(newtrafo, line);
			print(newtrafo, "(CompressResultImage true)");
			continue;
		}

		// change compression setting
		if (startsWith(line, "(FinalBSplineInterpolationOrder")) {
			// print(newtrafo, line);
			print(newtrafo, "(FinalBSplineInterpolationOrder 0)");
			continue;
		}

		print(newtrafo, line);
	}
	File.close(newtrafo);

	// upscale IF image on GPU
	open(moving_data);
	IF_img = getTitle();
	Ext.CLIJ2_push(IF_img);
	Ext.CLIJ2_getDimensions(IF_img, width_IF, height_IF, depth_IF);
	factor = width/width_IF;

	Ext.CLIJ2_downsample2D(IF_img, upscaled, factor, factor);
	Ext.CLIJ2_pull(upscaled);
	saveAs("tif", outdir + "tobetransformed");
	close();
	
	// Now, apply apply transform
	exec(transformix_exe + " " +  
		"-tp " + trafofile + " " +
		"-out "+ outdir + " " +
		"-in " + outdir + "tobetransformed.tif");
	File.delete(outdir + "tobetransformed.tif");

	// clean up
	trg = outdir + moving_img + "_transformed.tif";
	if (File.exists(trg)) {
		File.delete(trg);
	}
	File.rename(outdir + "result.tif", trg);

	if (File.exists(outdir + "result.0.tif")) {
		File.delete(outdir + "result.0.tif");
	}
	if (File.exists(outdir + "result.tif")) {
		File.delete(outdir + "result.tif");
	}
	return trg;
}

function calc_JC(f1, f2){
	open(f1);
	f1 = getTitle();

	open(f2);
	f2 = getTitle();

	Ext.CLIJ2_push(f1);
	Ext.CLIJ2_push(f2);

	close(f1);
	close(f2);

	Ext.CLIJ2_threshold(f1, img1, 1.0);
	Ext.CLIJ2_threshold(f2, img2, 1.0);

	Ext.CLIJ2_binaryAnd(img1, img2, Intersection);
	Ext.CLIJ2_binaryOr(img1, img2, Union);

	Ext.CLIJ2_getSumOfAllPixels(Intersection, I);
	Ext.CLIJ2_getSumOfAllPixels(Union, U);

	Ext.CLIJ2_clear();
	return (I/U);
}

function Mask(image, BG, roi){
	/*
	 * Input <image> will be masked according to the background value <BG>. 
	 * If <roi> is true, then the roi at index 0 in the roi manager will be used 
	 * to set the selected area to the background value <BG>.
	 */

	Ext.CLIJ2_clear();
	print("    ---> Moving " + image + " to GPU");
	Ext.CLIJ2_clear(); // empty GPU memory
	mask = "mask";
	tmp = "tmp";
	output = "output";
	selectWindow(image);

	// process background
	if (roi) {
		roiManager("select", 0);
		run("Set...", "value=" + BG);
	}

	// threshold
	Ext.CLIJ2_push(image);
	close(image);
	Ext.CLIJ2_convertUInt8(image, mask);
	Ext.CLIJ2_release(image);
	Ext.CLIJ2_labelToMask(mask, tmp, BG);
	Ext.CLIJ2_binaryNot(tmp, image);

	// Morphological post-processing
	N = 12;
	for (i = 0; i < N/2; i++) {
		Ext.CLIJ2_dilateSphere(image, tmp);
		Ext.CLIJ2_dilateSphere(tmp, image);
	}	

	// pull image for quick hole-filling
	Ext.CLIJ2_pull(image);
	image = getTitle();
	setThreshold(1, 255);
	run("Convert to Mask");
	run("Fill Holes");

	// push back and continue
	Ext.CLIJ2_push(image);
	Ext.CLIJ2_threshold(image, mask, 127);
	Ext.CLIJ2_release(image);
	Ext.CLIJ2_release(tmp);

	for (i = 0; i < N; i++) {
		Ext.CLIJ2_erodeSphere(mask, tmp);
		Ext.CLIJ2_erodeSphere(tmp, mask);
	}
	
	for (i = 0; i < N/2; i++) {
		Ext.CLIJ2_dilateSphere(mask, tmp);
		Ext.CLIJ2_dilateSphere(tmp, mask);
	}

	// calculate individual downsampling
	
	Ext.CLIJ2_getDimensions(mask, w, h, d);
	downsamplingfactor = w/registration_width;
	Ext.CLIJ2_downsample2D(mask, image, 1/downsamplingfactor, 1/downsamplingfactor);
	Ext.CLIJ2_multiplyImageAndScalar(image, mask, 255);
	close(image);
	
	Ext.CLIJ2_pull(image);
	print("    ---> Downsampled " + image + " by factor " + d2s(downsamplingfactor, 3));
	print("    ---> Fetching " + image + " to GPU");
	rename(image);
	
	// Lastly, pick largest blob from image
	// Prep
	roiManager("reset");
	setThreshold(1, 255);
	run("Analyze Particles...", "add");
	resetThreshold();
	Largest_area = 0;
	index = 0;

	run("Set Measurements...", "area redirect=None decimal=3");
	for (i = 0; i < roiManager("count"); i++) {
		roiManager("select", i);
		run("Measure");
		Area = getResult("Area", nResults - 1);
		if (Area > Largest_area) {
			index = i;
			Largest_area = Area;
		}
	}

	// Remove the small rest
	roiManager("select", index);
	run("Clear Outside");
	roiManager("reset");
	run("Select None");

	return downsamplingfactor;
}

function Load(f_input){
	// loads f_input
	//print("    --->"+f_input);

	open(f_input);
	rename(File.getNameWithoutExtension(f_input));
	print("    --> Loaded" + f_input);
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
