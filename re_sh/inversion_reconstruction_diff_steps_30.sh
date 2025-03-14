CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a building with a facade that appears to be made of stone or brick, with a white mortar or plaster finish. The building has a series of windows with red shutters, and there is a red door on the ground floor. The windows are evenly spaced, and the shutters are closed. The building looks old and possibly historic, given the style of the windows and the material of the facade. There are some plants growing at the base of the building, adding a touch of greenery to the scene. The overall impression is of a quaint, possibly residential building." \
                --target_prompt "The image shows a building with a facade that appears to be made of stone or brick, with a white mortar or plaster finish. The building has a series of windows with red shutters, and there is a red door on the ground floor. The windows are evenly spaced, and the shutters are closed. The building looks old and possibly historic, given the style of the windows and the material of the facade. There are some plants growing at the base of the building, adding a touch of greenery to the scene. The overall impression is of a quaint, possibly residential building." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim01.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim01' \
                --output_dir 'examples/edit-result/reconstruction/step30' 

CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a red wooden door with a metal latch and a metal bar attached to it. There is a round object, which appears to be a stone or ceramic ball, hanging from the latch. The door has a rustic appearance, with visible wood grain and some wear, suggesting it might be an old or weathered door. The bar is likely used for additional security or to prevent the door from being accidentally opened. The ball could serve as a decorative element or might have a functional purpose, such as to prevent the latch from being accidentally tripped." \
                --target_prompt "The image shows a red wooden door with a metal latch and a metal bar attached to it. There is a round object, which appears to be a stone or ceramic ball, hanging from the latch. The door has a rustic appearance, with visible wood grain and some wear, suggesting it might be an old or weathered door. The bar is likely used for additional security or to prevent the door from being accidentally opened. The ball could serve as a decorative element or might have a functional purpose, such as to prevent the latch from being accidentally tripped." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim02.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim02' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a row of five baseball caps hanging on a wall or a similar surface. Each cap has a different color and a label with the word \"Boca\" printed on it. The colors of the caps, from left to right, are yellow, orange, green, pink, and blue. The caps are arranged in a line, and the sunlight casts shadows on the wall, indicating that the photo was taken on a sunny day." \
                --target_prompt "The image shows a row of five baseball caps hanging on a wall or a similar surface. Each cap has a different color and a label with the word \"Boca\" printed on it. The colors of the caps, from left to right, are yellow, orange, green, pink, and blue. The caps are arranged in a line, and the sunlight casts shadows on the wall, indicating that the photo was taken on a sunny day." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim03.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim03' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a person wearing a red hat with a mesh pattern, a red scarf, and a necklace with a white and black design. The person has blonde hair and is smiling at the camera. The background is a soft pink color, which gives the image a warm and vintage feel. The style of the image suggests it could be from the 1980s or 1990s, a time when such fashion was popular." \
                --target_prompt "The image shows a person wearing a red hat with a mesh pattern, a red scarf, and a necklace with a white and black design. The person has blonde hair and is smiling at the camera. The background is a soft pink color, which gives the image a warm and vintage feel. The style of the image suggests it could be from the 1980s or 1990s, a time when such fashion was popular." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim04.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim04' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a group of motocross riders lined up on their bikes, likely preparing for a race or event. They are wearing full motocross gear, including helmets, goggles, gloves, and protective clothing. The bikes are designed for off-road racing, with knobby tires for traction on dirt surfaces. The riders are positioned in a way that suggests they are either waiting for the start of the race or have just finished a race and are waiting for their turn to cross the finish line. The environment looks like a motocross track, with a dirt surface and a grassy area in the background." \
                --target_prompt "The image shows a group of motocross riders lined up on their bikes, likely preparing for a race or event. They are wearing full motocross gear, including helmets, goggles, gloves, and protective clothing. The bikes are designed for off-road racing, with knobby tires for traction on dirt surfaces. The riders are positioned in a way that suggests they are either waiting for the start of the race or have just finished a race and are waiting for their turn to cross the finish line. The environment looks like a motocross track, with a dirt surface and a grassy area in the background." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim05.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim05' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a small sailboat floating on a body of water, likely the ocean, given the presence of waves and the horizon line. The boat is equipped with a mast and a sail, which is not currently being used, suggesting it might be a recreational or fishing vessel. The name \"ZENTIME\" is visible on the side of the boat, indicating its name or registration. The sky is partly cloudy, and the lighting suggests it could be either early morning or late afternoon, given the warm tones and the angle of the sunlight. There are no people visible on the boat, and the water appears calm."\
                --target_prompt "The image shows a small sailboat floating on a body of water, likely the ocean, given the presence of waves and the horizon line. The boat is equipped with a mast and a sail, which is not currently being used, suggesting it might be a recreational or fishing vessel. The name \"ZENTIME\" is visible on the side of the boat, indicating its name or registration. The sky is partly cloudy, and the lighting suggests it could be either early morning or late afternoon, given the warm tones and the angle of the sunlight. There are no people visible on the boat, and the water appears calm."\
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim06.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim06' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a pink flower with a prominent red center, which appears to be a type of hibiscus, hanging from a branch. The flower is in focus, with a shallow depth of field that blurs the background, which includes a window with blue shutters and a white wall. There is also a red flower in the foreground, adding to the vibrant colors in the scene. The overall setting suggests a warm, possibly tropical, environment." \
                --target_prompt "The image shows a pink flower with a prominent red center, which appears to be a type of hibiscus, hanging from a branch. The flower is in focus, with a shallow depth of field that blurs the background, which includes a window with blue shutters and a white wall. There is also a red flower in the foreground, adding to the vibrant colors in the scene. The overall setting suggests a warm, possibly tropical, environment." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim07.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim07' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a row of traditional European-style houses with steeply pitched roofs and dormer windows. The architecture suggests a location in a region with a history of such buildings, possibly in a town or village in Europe. The buildings are closely packed together, which is typical of historic European towns where space was limited. The facades are colorful, with a mix of red, yellow, and green, which is also characteristic of certain regions in Europe. The presence of a street lamp and the style of the buildings suggest this could be a scene from a town in Germany, such as Rothenburg ob der Tauber, which is famous for its half-timbered houses." \
                --target_prompt "The image shows a row of traditional European-style houses with steeply pitched roofs and dormer windows. The architecture suggests a location in a region with a history of such buildings, possibly in a town or village in Europe. The buildings are closely packed together, which is typical of historic European towns where space was limited. The facades are colorful, with a mix of red, yellow, and green, which is also characteristic of certain regions in Europe. The presence of a street lamp and the style of the buildings suggest this could be a scene from a town in Germany, such as Rothenburg ob der Tauber, which is famous for its half-timbered houses." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim08.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim08' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a group of sailboats on a body of water, likely participating in a sailing event or race. The sails are colorful, with various patterns and designs, and the boats are numbered, indicating that they are part of a competition. The sky is overcast, suggesting it might be a cool or cloudy day. The sailors are wearing life jackets, which is a safety precaution in such activities. The water appears calm, which is typical for sailing conditions." \
                --target_prompt "The image shows a group of sailboats on a body of water, likely participating in a sailing event or race. The sails are colorful, with various patterns and designs, and the boats are numbered, indicating that they are part of a competition. The sky is overcast, suggesting it might be a cool or cloudy day. The sailors are wearing life jackets, which is a safety precaution in such activities. The water appears calm, which is typical for sailing conditions." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim09.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim09' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a sailing scene with several sailboats on the water. The sailboats are equipped with large sails, and there are people on board, likely enjoying a day of sailing. The sails are up, catching the wind, and the boats are moving across the water. The sky is clear, suggesting good weather conditions for sailing. The boats are numbered, indicating that this might be a race or a regatta, where each boat has a unique identification number." \
                --target_prompt "The image shows a sailing scene with several sailboats on the water. The sailboats are equipped with large sails, and there are people on board, likely enjoying a day of sailing. The sails are up, catching the wind, and the boats are moving across the water. The sky is clear, suggesting good weather conditions for sailing. The boats are numbered, indicating that this might be a race or a regatta, where each boat has a unique identification number." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim10.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim10' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a small boat moored at a wooden dock. The boat appears to be a simple, possibly traditional design, with a white hull and a red interior. The dock looks weathered and somewhat dilapidated, suggesting it might be in a location that experiences harsh weather conditions or is not well-maintained. The water around the boat is calm, and the sky is overcast, indicating it might be a cloudy day. There are no people visible in the image, and the overall scene gives a sense of tranquility and solitude." \
                --target_prompt "The image shows a small boat moored at a wooden dock. The boat appears to be a simple, possibly traditional design, with a white hull and a red interior. The dock looks weathered and somewhat dilapidated, suggesting it might be in a location that experiences harsh weather conditions or is not well-maintained. The water around the boat is calm, and the sky is overcast, indicating it might be a cloudy day. There are no people visible in the image, and the overall scene gives a sense of tranquility and solitude."\
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim11.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim11' \
                --output_dir 'examples/edit-result/reconstruction/step30' 

CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows two people walking through shallow water on a beach. They appear to be enjoying a day at the beach, possibly preparing for or returning from water activities such as snorkeling or swimming. The water is calm, and the beach looks serene with a few trees in the background. The sky is partly cloudy, suggesting a pleasant day for outdoor activities."\
                --target_prompt "The image shows two people walking through shallow water on a beach. They appear to be enjoying a day at the beach, possibly preparing for or returning from water activities such as snorkeling or swimming. The water is calm, and the beach looks serene with a few trees in the background. The sky is partly cloudy, suggesting a pleasant day for outdoor activities." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim12.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim12' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a mountainous landscape with a river flowing through it. The river is surrounded by a dense forest of coniferous trees, and there are rocky areas along the riverbank. In the background, there are snow-capped mountains, suggesting a high-altitude environment. The sky is partly cloudy, indicating a mix of sunny and overcast conditions. The overall scene is typical of a mountainous region, possibly in a temperate or subalpine climate." \
                --target_prompt "The image shows a mountainous landscape with a river flowing through it. The river is surrounded by a dense forest of coniferous trees, and there are rocky areas along the riverbank. In the background, there are snow-capped mountains, suggesting a high-altitude environment. The sky is partly cloudy, indicating a mix of sunny and overcast conditions. The overall scene is typical of a mountainous region, possibly in a temperate or subalpine climate."\
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim13.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim13' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a group of people on a yellow inflatable raft, navigating a river with rapids. They are wearing life jackets for safety, and some are holding paddles to help steer the raft. The water appears to be rough, indicating that they are in a section of the river with challenging rapids. The raft is labeled \"SNOWMASS WHITEWATER,\" which suggests that this is a specific location known for white-water rafting, likely in the Snowmass area of Colorado, which is a popular destination for such activities. The people seem to be enjoying the adventure, and the environment looks beautiful with the surrounding rocks and the dynamic water." \
                --target_prompt "The image shows a group of people on a yellow inflatable raft, navigating a river with rapids. They are wearing life jackets for safety, and some are holding paddles to help steer the raft. The water appears to be rough, indicating that they are in a section of the river with challenging rapids. The raft is labeled \"SNOWMASS WHITEWATER,\" which suggests that this is a specific location known for white-water rafting, likely in the Snowmass area of Colorado, which is a popular destination for such activities. The people seem to be enjoying the adventure, and the environment looks beautiful with the surrounding rocks and the dynamic water." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim14.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim14' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a young child with a colorful face paint design. The paint includes a large yellow circle around one eye, with additional splashes of blue, red, and white around the face. The child is wearing a colorful, patterned garment with a red and green color scheme. The child is looking directly at the camera with a neutral expression. The background is plain and light-colored, which helps to focus on the child and the face paint." \
                --target_prompt "The image shows a young child with a colorful face paint design. The paint includes a large yellow circle around one eye, with additional splashes of blue, red, and white around the face. The child is wearing a colorful, patterned garment with a red and green color scheme. The child is looking directly at the camera with a neutral expression. The background is plain and light-colored, which helps to focus on the child and the face paint." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim15.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim15' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a tropical island with a sandy beach and a line of palm trees. The sky is partly cloudy, with some clouds appearing to be cumulus, which are often associated with fair weather. The water in the foreground reflects the sky and the island, creating a serene and picturesque scene. The island is surrounded by a body of water, which could be an ocean or a large lake, given the horizon line. The overall atmosphere of the image is tranquil and suggests a location that might be a popular destination for relaxation and vacation." \
                --target_prompt  "The image shows a tropical island with a sandy beach and a line of palm trees. The sky is partly cloudy, with some clouds appearing to be cumulus, which are often associated with fair weather. The water in the foreground reflects the sky and the island, creating a serene and picturesque scene. The island is surrounded by a body of water, which could be an ocean or a large lake, given the horizon line. The overall atmosphere of the image is tranquil and suggests a location that might be a popular destination for relaxation and vacation." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim16.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim16' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt"The image shows a statue of a figure that appears to be inspired by classical or mythological themes. The figure is adorned with a crown of leaves and is holding what looks like a large, round, golden object, possibly a ball or an egg. The statue is depicted with a serene expression and is draped in a flowing garment, which suggests a sense of majesty or divinity. The setting appears to be outdoors, with natural light illuminating the statue, enhancing its details and the textures of the materials used." \
                --target_prompt "The image shows a statue of a figure that appears to be inspired by classical or mythological themes. The figure is adorned with a crown of leaves and is holding what looks like a large, round, golden object, possibly a ball or an egg. The statue is depicted with a serene expression and is draped in a flowing garment, which suggests a sense of majesty or divinity. The setting appears to be outdoors, with natural light illuminating the statue, enhancing its details and the textures of the materials used."\
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim17.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim17' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a person standing in front of a large, rusted metal sculpture. The sculpture appears to be a representation of a bird or a winged creature, with a prominent, curved shape that could be interpreted as a wing or a tail. The person is wearing a dark dress with a pearl necklace and has long hair. They are posing with one hand on the sculpture, and the background suggests a natural setting with greenery. The overall composition of the image suggests a connection between the person and the artwork, possibly for a photoshoot or an artistic project."\
                --target_prompt "The image shows a person standing in front of a large, rusted metal sculpture. The sculpture appears to be a representation of a bird or a winged creature, with a prominent, curved shape that could be interpreted as a wing or a tail. The person is wearing a dark dress with a pearl necklace and has long hair. They are posing with one hand on the sculpture, and the background suggests a natural setting with greenery. The overall composition of the image suggests a connection between the person and the artwork, possibly for a photoshoot or an artistic project." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim18.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim18' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a lighthouse with a distinctive white tower and a black lantern room at the top. The lighthouse is situated on a grassy area, and there is a white picket fence in front of it. To the left of the lighthouse, there is a white building with a gray roof, and a red lifebuoy is attached to the fence. The sky is partly cloudy, suggesting it might be a cool or overcast day. The lighthouse appears to be a historic structure, possibly serving as a navigational aid for maritime pilots." \
                --target_prompt "The image shows a lighthouse with a distinctive white tower and a black lantern room at the top. The lighthouse is situated on a grassy area, and there is a white picket fence in front of it. To the left of the lighthouse, there is a white building with a gray roof, and a red lifebuoy is attached to the fence. The sky is partly cloudy, suggesting it might be a cool or overcast day. The lighthouse appears to be a historic structure, possibly serving as a navigational aid for maritime pilots." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim19.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim19' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a vintage propeller-driven aircraft, likely a fighter or a fighter-bomber, parked on a grassy field. The aircraft has a distinctive color scheme with a combination of yellow, black, and white stripes on the wings and tail. The text \"Sky Shooter\" is visible on the side of the aircraft, which suggests it might be a custom or personal name for the plane. The propeller is clearly visible, and the aircraft appears to be in good condition, indicating it may be a well-maintained or restored vintage aircraft. The background is somewhat blurred, but it looks like an open field with a clear sky, which is typical for an airfield or a location where such aircraft are displayed or operated." \
                --target_prompt "The image shows a vintage propeller-driven aircraft, likely a fighter or a fighter-bomber, parked on a grassy field. The aircraft has a distinctive color scheme with a combination of yellow, black, and white stripes on the wings and tail. The text \"Sky Shooter\" is visible on the side of the aircraft, which suggests it might be a custom or personal name for the plane. The propeller is clearly visible, and the aircraft appears to be in good condition, indicating it may be a well-maintained or restored vintage aircraft. The background is somewhat blurred, but it looks like an open field with a clear sky, which is typical for an airfield or a location where such aircraft are displayed or operated." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim20.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim20' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows a lighthouse situated on a rocky outcrop near the ocean. The lighthouse is tall and white with a black lantern room at the top, which is characteristic of many lighthouses designed to serve as navigational aids for maritime pilots. In the foreground, there is a building with a red roof, which appears to be a house or a small structure, possibly related to the operation of the lighthouse. The sky is partly cloudy, and the ocean is visible in the background, suggesting a coastal location. The overall scene is picturesque and typical of coastal areas where lighthouses are often found to guide ships and warn of potential hazards."\
                --target_prompt "The image shows a lighthouse situated on a rocky outcrop near the ocean. The lighthouse is tall and white with a black lantern room at the top, which is characteristic of many lighthouses designed to serve as navigational aids for maritime pilots. In the foreground, there is a building with a red roof, which appears to be a house or a small structure, possibly related to the operation of the lighthouse. The sky is partly cloudy, and the ocean is visible in the background, suggesting a coastal location. The overall scene is picturesque and typical of coastal areas where lighthouses are often found to guide ships and warn of potential hazards." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim21.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim21' \
                --output_dir 'examples/edit-result/reconstruction/step30' 

CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a red barn with a gabled roof, situated in a rural setting. In front of the barn, there is a small body of water, possibly a pond or a small lake, reflecting the barn's colors and structure. The surrounding area is lush with green grass and trees, suggesting a peaceful, agricultural landscape. The sky is overcast, indicating that the photo was taken on a cloudy day." \
                --target_prompt  "The image shows a red barn with a gabled roof, situated in a rural setting. In front of the barn, there is a small body of water, possibly a pond or a small lake, reflecting the barn's colors and structure. The surrounding area is lush with green grass and trees, suggesting a peaceful, agricultural landscape. The sky is overcast, indicating that the photo was taken on a cloudy day." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim22.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim22' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt "The image shows two macaws, which are large, colorful parrots known for their vibrant plumage and strong, curved beaks. The bird on the left has a predominantly green head with a yellow body and blue wings, while the bird on the right has a red head with a white face, a red body, and blue wings. Both birds have distinctive markings around their eyes and beaks, which are characteristic of their species. They appear to be in a natural setting with green foliage in the background, suggesting they might be in a tropical or subtropical environment."\
                --target_prompt "The image shows two macaws, which are large, colorful parrots known for their vibrant plumage and strong, curved beaks. The bird on the left has a predominantly green head with a yellow body and blue wings, while the bird on the right has a red head with a white face, a red body, and blue wings. Both birds have distinctive markings around their eyes and beaks, which are characteristic of their species. They appear to be in a natural setting with green foliage in the background, suggesting they might be in a tropical or subtropical environment."\
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim23.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim23' \
                --output_dir 'examples/edit-result/reconstruction/step30' 
CUDA_VISIBLE_DEVICES=0 python edit.py  --source_prompt  "The image shows a traditional European-style house with a colorful and detailed facade. The house features a variety of scenes and figures painted on its exterior walls, which is a common decorative style in certain regions of Europe, particularly in the Alps. The architecture includes a gabled roof with a red tile covering, and there are balconies with ornate railings. The house is surrounded by lush greenery, and there are potted plants placed on the ground in front of the house, adding to the picturesque setting. The house appears to be situated in a mountainous area, as suggested by the background that includes trees and what looks like a mountain slope." \ 
                --target_prompt "The image shows a traditional European-style house with a colorful and detailed facade. The house features a variety of scenes and figures painted on its exterior walls, which is a common decorative style in certain regions of Europe, particularly in the Alps. The architecture includes a gabled roof with a red tile covering, and there are balconies with ornate railings. The house is surrounded by lush greenery, and there are potted plants placed on the ground in front of the house, adding to the picturesque setting. The house appears to be situated in a mountainous area, as suggested by the background that includes trees and what looks like a mountain slope." \
                --guidance 1 \
                --source_img_dir '/home/ailab/ailab_datasets/Kodak/kodim24.png' \
                --num_steps 30 \
                --offload \
                --inject 0 \
                --sampling_strategy 'fireflow' \
                --output_prefix 'fireflow_reconstruction_kodim24' \
                --output_dir 'examples/edit-result/reconstruction/step30' 

