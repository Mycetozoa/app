06/09/2017 19:35:34 Schnittler Martin  
University of Greifswald, Institute of Botany and Landscape Ecology (Germany)  
mschnitt@uni-greifswald.de  
Automated spore ornament pattern analysis  
Spores are unicellular, airborne propagules of different organisms (from plants and fungi to protists) usually 5-50 µm in size, spherical to ellipsoid, and their surface is covered with various ornaments, which consist of warts, spines, ridges and all combinations of these.  
These spore ornaments often vary between species. For fungi and slime molds, spore ornamentation is one of the most distinctive characters for species differentiation.  
Distinguishing the spore ornaments is so far the domain of skilled taxonomists – you need a good eye to memorize and tell apart the manifold varieties of such ornaments. Surprisingly, a good quantitative approach to describe these ornaments is missing – therefore in species descriptions one finds usually wonderful images of spores, often with scanning electron microscopy (SEM), but not more. A quantitative approach would enable us to use the parameters describing spore ornamentation in all kinds of analyses, like multivariate statistics, numerical taxonomy, machine-guided identification etc. Thus, we would like to develop a method for an automated quantitative analysis of spore ornamentation patterns from SEM micrographs done with the critical point drying.   

ImageJ (U.S. National Institute of Health) is a freeware software that allows flexible automated batch processing of digital images using scripts. Our current pipeline of SEM micrographs processing in ImageJ faces three major problems that should be solved:
1) Only a small part of a spore can be analysed since only a minor part of the spore surface (in exact mathematical terms only a point) is directly seen from above.   
Task to do: produce a true-distance projection of the spore ornaments for spherical spores of slime molds. The interpolated, projected section of the spore may then be used for further analysis.  
2) When converting from grayscale to black and white images, not all the ornaments can be separated from the background with global separation threshold. Moreover, so far the threshold for the separation is not derived automatically, but must be found by the user.   
Task to do: an automated separate calculation of thresholds for different sections.   
3) Sunken ornaments:in the case of rather shallow ornaments, sometimes only the outer crests appear in the b/w images, not the entire ornaments.   
Task to do: an algorithm must be found to connect missing parts at the outlines and fill the ornament.    

The expected result is a script for batch image processing in ImageJ or any other software chosen by the student.   
