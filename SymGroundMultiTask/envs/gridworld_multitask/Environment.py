import gym
from gym import spaces
import random
import numpy as np
import torch
import cv2
import os




ENV_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(ENV_DIR))


class GridWorldEnv_multitask(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array", "terminal"],
        "state_types": ["image", "symbol"],
        "render_fps": 4
    }

    symbol_to_name = {
        'c0': 'pick', 'c1': 'lava', 'c2': 'door', 'c3': 'gem', 'c4': 'egg', 'c5': 'sword',
        'c6': 'bow', 'c7': 'arrow', 'c8': 'potion', 'c9': 'book', 'c10': 'food', 'c11': 'gold',
        '': 'nothing'
    }

    symbol_to_index = { symbol: idx for idx, symbol in enumerate(symbol_to_name.keys()) }


    

    def __init__(self, render_mode="human", state_type="image", obs_size=(56,56), win_size=(896,896), map_size=7,
        max_num_steps=30, randomize_loc=True, randomize_start=True, img_dir="imgs_16x16", save_obs=False,
        symbols=['c0', 'c1', 'c2', 'c3'], wrap_around_map=True, agent_centric_view=True):

        self.dictionary_symbols = symbols + ['']
        self.num_symbols = len(self.dictionary_symbols)

        print(max_num_steps)
        
        
        
        
        GridWorldEnv_multitask.symbol_to_index = { symbol: idx for idx, symbol in enumerate(self.dictionary_symbols) }

        
        #the selected symbols for the current environment
        print("Symbol to name mapping:")
        for symbol in self.dictionary_symbols:
            print(f"  {symbol} -> {self.symbol_to_name[symbol]}")

        

        
        self.dictionary_icons = [self.symbol_to_name[symbol] for symbol in self.dictionary_symbols[:-1]]
        
        
        self.randomize_locations = randomize_loc
        self.randomize_start = randomize_start

        self.max_num_steps = max_num_steps
        self.curr_step = 0
        self.num_episodes = 0

        self.has_window = False
        self.map_size = map_size
        self.obs_size = obs_size

        self.save_obs = save_obs
        
        
        
        
        self.win_size = win_size

        assert not agent_centric_view or (self.map_size%2==1 and wrap_around_map)
        assert self.num_symbols*2 <= self.map_size**2
        self.wrap_around_map = wrap_around_map
        self.agent_centric_view = agent_centric_view

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert state_type in self.metadata["state_types"]
        self.state_type = state_type

        self.action_space = spaces.Discrete(4)
        self.action_to_direction = {
            0: (0, 1),  # DOWN
            1: (1, 0),  # RIGHT
            2: (0, -1),  # UP
            3: (-1, 0),  # LEFT
        }

        # variables to hide icons
        self.agent_display = True

       

        # load icons using OpenCV (if they are used)
        if self.state_type == 'image' or self.render_mode in ['human', 'rgb_array']:

            # icons file paths
            self.icons_paths = {}
            for icon in self.dictionary_icons:
                self.icons_paths[icon] = os.path.join(ENV_DIR, img_dir, icon+".png")
            self.icons_paths['agent'] = os.path.join(ENV_DIR, img_dir, "agent.png")

            # icons files
            self.icons = {}
            for icon in self.dictionary_icons:
                self.icons[icon] = cv2.imread(self.icons_paths[icon], cv2.IMREAD_UNCHANGED)
            self.icons['agent'] = cv2.imread(self.icons_paths['agent'], cv2.IMREAD_UNCHANGED)

            # dimensions of true canvas
            self.cell_size = self.icons['pick'].shape[0]
            self.canvas_size = self.map_size * self.cell_size

        # default locations
        self.all_locations = {(x,y) for x in range(self.map_size) for y in range(self.map_size)}
        default_locations = [(1,1), (5,2), (3,3), (1,4), (3,0), (3,5), (0,3), (6,4), (2,1), (5,6)]
        
        default_locations = [(0,0), (0,3), (0,6), (1,1), (1,4), (1,6), (2,1), (2,5), (3,0), (3,3),
                             (3,5), (4,1), (4,4), (4,6), (5,0), (5,2), (5,6), (6,1), (6,3), (6,5),
                             (2,3), (3,6), (6,0), (0,5)]

        default_locations = default_locations[:(self.num_symbols-1)*2]

        # assign items locations
        self.icon_locations = {}
        for label, icon in enumerate(self.dictionary_icons):
            self.icon_locations[icon] = default_locations[label*2:(label+1)*2]

        # assign agent initial location
        self.free_locations = self.all_locations - set(default_locations)
        self.initial_agent_location = (0,0)

        self.loc_to_labels = {}
        self.loc_to_obs = {}
        self._precompute_observations(save_obs)

        if self.state_type == "image":
            self.observation_space = spaces.Box(
                low = np.float32(-np.inf),
                high = np.float32(np.inf),
                shape = self.loc_to_obs[0,0].shape,
                dtype = np.float32
            )

        elif self.state_type == "symbol":
            self.observation_space = spaces.Box(low=0, high=1, shape=self.loc_to_obs[0,0].shape, dtype=np.uint8)

        # reset the agent location
        self.agent_location = self.initial_agent_location


    def reset(self):  	

        self.num_episodes += 1
        self.curr_step = 0

        # randomize item locations and recompute observations
        if self.randomize_locations:

            # select random locations for items
            num_items = (self.num_symbols-1)*2
            #num_items = self.num_symbols - 1
            sampled_locations = random.sample(self.all_locations, num_items)

            # assign item locations
            for label, icon in enumerate(self.dictionary_icons):
                self.icon_locations[icon] = sampled_locations[label*2:(label+1)*2]

            # assign agent initial location
            self.free_locations = self.all_locations - set(sampled_locations)
            self.initial_agent_location = random.sample(self.free_locations, 1)[0]

            self._precompute_observations()

        # extract new initial location
        elif self.randomize_start:
            self.initial_agent_location = random.sample(self.free_locations, 1)[0]

        # reset the agent location
        self.agent_location = self.initial_agent_location

        # compute initial observation
        observation = self.loc_to_obs[self.agent_location]

        #observation = self._get_image_obs()

        if self.save_obs:

            # #save as an image the observation
            image = (np.transpose(observation * 255)).astype(np.uint8)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # #save it in ./
            obs_folder = os.path.join(REPO_DIR, 'saves/env_obs')
            if not os.path.exists(obs_folder):
                os.makedirs(obs_folder)
            obs_path = os.path.join(obs_folder, f'obs_reset_{self.num_episodes}.png')
            cv2.imwrite(obs_path, image_bgr)
            print(f"Saved observation image at {obs_path}")
        



        return observation


    def step(self, action):

        self.curr_step += 1
        direction = self.action_to_direction[action]

        # update agent location
        if self.wrap_around_map:
            self.agent_location = ((self.agent_location[0] + direction[0]) % self.map_size,
                                   (self.agent_location[1] + direction[1]) % self.map_size)
        else:
            self.agent_location = (max(0, min(self.map_size-1, self.agent_location[0] + direction[0])),
                                   max(0, min(self.map_size-1, self.agent_location[1] + direction[1])))

        # compute values to return
        observation = self.loc_to_obs[self.agent_location].copy()

        
        reward = 0.0
        done = self.curr_step >= self.max_num_steps
        info = None

        # #DEBUG: save the observation as an image and exit
        # debug_folder = os.path.join(os.getcwd(), 'Debug')
        # print("Debug folder:", debug_folder)

        # if not os.path.exists(debug_folder):
        #     os.makedirs(debug_folder)

        # print(observation)
        # exit()


        # image = (np.transpose(observation *  (self.stdev + 1e-10) + self.mean, (1, 2, 0)) * 255).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        # obs_path = os.path.join(debug_folder, f'obs_step_{self.num_episodes}_{self.curr_step}.png')
        # cv2.imwrite(obs_path, image)
        # print(f"Saved observation image at {obs_path}")

        # print(image)


        
        
        

        return observation, reward, done, info


    def set_map(self, map_dict):

        assert map_dict.keys() == self.dictionary_icons + ['agent']
        used_locations = (map_dict[icon] for icon in self.dictionary_icons)

        assert len(used_locations + [map_dict['agent']]) == len(set(used_locations + [map_dict['agent']]))
        for pos in (used_locations + [map_dict['agent']]):
            assert pos in self.all_locations

        for icon in self.dictionary_icons:
            self.icon_locations[icon] = map_dict[icon]

        self.free_locations = self.all_locations - set(used_locations)
        self.initial_agent_location = map_dict['agent']

        self._precompute_observations()
        self.agent_location = self.initial_agent_location

        return self.loc_to_obs[self.agent_location]


    def _precompute_observations(self, save_obs=False):
        save_obs = False
        # precompute symbols per location
        self.loc_to_label = {loc: self.num_symbols-1 for loc in self.all_locations}
        for label, icon in enumerate(self.dictionary_icons):
            for loc in self.icon_locations[icon]:
                self.loc_to_label[loc] = label
                

        if self.state_type == "image":

            # precompute image observations per location
            self.loc_to_obs = {}
            for loc in self.all_locations:
                self.agent_location = loc
                self.loc_to_obs[loc] = self._get_image_obs()

            # save images as seen by the agent
            if save_obs:
                obs_folder = os.path.join(REPO_DIR, 'saves/env_obs')
                if not os.path.exists(obs_folder):
                    os.makedirs(obs_folder)
                for r,c in self.all_locations:
                    image = (np.transpose(self.loc_to_obs[r,c], (1, 2, 0)) * 255).astype(np.uint8)
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    obs_path = os.path.join(obs_folder, f'obs_{r}_{c}.png')
                    
                    cv2.imwrite(obs_path, image_bgr)

            # normalize observations
            #mean = np.mean(self.loc_to_obs[self.initial_agent_location])
            #stdev = np.std(self.loc_to_obs[self.initial_agent_location])


            #self.mean = mean
            #self.stdev = stdev

            #print(f"Observation mean: {mean}, stdev: {stdev}")


            # for loc in self.all_locations:
            #     norm_img = (self.loc_to_obs[loc] - mean) / (stdev + 1e-10)
            #     self.loc_to_obs[loc] = norm_img
            #     if save_obs:
            #         image = (np.transpose(norm_img, (1, 2, 0)) * 255).astype(np.uint8)
            #         image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #         obs_path = os.path.join(obs_folder, f'obs_norm_{loc[0]}_{loc[1]}.png')
            #         cv2.imwrite(obs_path, image_bgr)

        elif self.state_type == "symbol":

            # precompute symbol observations per location
            self.loc_to_obs = {}
            for loc in self.all_locations:
                self.agent_location = loc
                self.loc_to_obs[loc] = self._get_symbol_obs()


    def _get_symbol_obs(self):

        obs = np.zeros(shape=(self.map_size,self.map_size,self.num_symbols),dtype=np.uint8)

        for loc in self.all_locations:
            label = self.loc_to_label[loc]
            if self.agent_centric_view:
                loc = self._absolute_to_agent_centric(loc)
            if label != self.num_symbols-1:
                obs[loc[0],loc[1],label] = 1

        loc = self.agent_location
        if self.agent_centric_view:
            loc = self._absolute_to_agent_centric(loc)
        obs[loc[0],loc[1],self.num_symbols-1] = 1

        return obs


    def _get_image_obs(self):
        obs = self._render_frame()
        obs = cv2.resize(obs, self.obs_size)
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (2, 0, 1)) # from w*h*c to c*w*h
        return obs


    def _get_info(self):
        info = {
            'agent location': self.agent_location,
        }
        return info


    # create the visualization of the environment (for rendering and for observations)
    def _render_frame(self, draw_grid=False):

        # create a white canvas
        canvas = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)

        # draw grid lines
        if draw_grid:
            positions = range(0, self.canvas_size + 1, self.cell_size)
            for i in positions:
                cv2.line(canvas, (0, i), (self.canvas_size, i), (0, 0, 0), 3)
                cv2.line(canvas, (i, 0), (i, self.canvas_size), (0, 0, 0), 3)

        # helper function to overlay an image with transparency if available
        def overlay_image(bg, fg, top_left):
            x, y = top_left
            h, w = fg.shape[:2]
            if fg.shape[2] == 4:
                alpha_fg = fg[:, :, 3:] / 255.0
                alpha_bg = 1.0 - alpha_fg
                bg_slice = bg[y:y+h, x:x+w]
                fg_rgb = fg[:, :, :3]
                blended = alpha_fg * fg_rgb + alpha_bg * bg_slice
                bg[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                bg[y:y+h, x:x+w] = fg
            return bg

        # calculate pixel positions for each grid item
        def blit_item(item_img, locations, display=True):
            if display:
                for loc in locations:
                    if self.agent_centric_view:
                        loc = self._absolute_to_agent_centric(loc)
                    x = int(loc[0] * self.cell_size)
                    y = int(loc[1] * self.cell_size)
                    overlay_image(canvas, item_img, (x, y))

        # blit each type of item
        for icon in self.dictionary_icons:
            blit_item(self.icons[icon], self.icon_locations[icon])
        blit_item(self.icons['agent'], [self.agent_location], self.agent_display)

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


    def _absolute_to_agent_centric(self, pos):
        center = self.map_size // 2
        delta  = (center - self.agent_location[0], center - self.agent_location[1])
        new_pos_r = (pos[0] + delta[0] + self.map_size) % self.map_size
        new_pos_c = (pos[1] + delta[1] + self.map_size) % self.map_size
        return (new_pos_r, new_pos_c)


    def _agent_centric_to_absolute(self, pos):
        center = self.map_size // 2
        delta  = (center - self.agent_location[0], center - self.agent_location[1])
        orig_r = (pos[0] - delta[0] + self.map_size) % self.map_size
        orig_c = (pos[1] - delta[1] + self.map_size) % self.map_size
        return (orig_r, orig_c)


    def render(self):
        if self.render_mode == "human":
            return self.show()
        elif self.render_mode == "rgb_array":
            return self.loc_to_obs[self.agent_location]
        elif self.render_mode == "terminal":
            return self.show_to_terminal()


    def translate_formula(self, formula):

        if isinstance(formula, tuple):
            return tuple(self.translate_formula(item) for item in formula)
        elif formula in self.symbol_to_name:
            return self.symbol_to_name[formula]
        else:
            return formula


    def show_to_terminal(self):

        for c in range(self.map_size):
            row_str = ""
            for r in range(self.map_size):
                loc = (r, c)
                if self.agent_centric_view:
                    loc = self._agent_centric_to_absolute(loc)
                label = self.loc_to_label.get(loc, self.num_symbols-1)
                letter = self.dictionary_symbols[label]
                if self.agent_location == loc:
                    row_str += f"[{letter}]"
                else:
                    row_str += f" {letter} "
            print(row_str)


    def show(self):
        if not self.has_window:
            self.has_window = True
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame", self.win_size[0], self.win_size[1])
            cv2.moveWindow('Frame', 100, 100)
        canvas = cv2.cvtColor(self._render_frame(), cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, self.win_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Frame", canvas)
        cv2.waitKey(1)


    def close(self):
        if self.has_window:
            self.has_window = False
            cv2.destroyWindow("Frame")



# interface needed to build the ltl_wrapper
# the agent doesn't receive the current symbol from the environment but computes 
# it through its grounder model, which is attached to the environment.
class GridWorldEnv_LTL2Action(GridWorldEnv_multitask):

    def __init__(self, grounder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_grounder = grounder
        self.current_obs = None
        assert self.state_type == 'image' or self.sym_grounder == None


    def reset(self):
        obs = super().reset()
        self.current_obs = obs
        return obs


    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.current_obs = obs
        #da la nostra step dal vecchio env
        return obs, rew, done, info


    def get_propositions(self):
        return self.dictionary_symbols[:-1].copy()


    def get_real_events(self):
        real_sym = self.loc_to_label[self.agent_location]
        return self.dictionary_symbols[real_sym]


    def get_events(self):

        # returns the proposition that currently holds
        if self.sym_grounder == None:
            return self.get_real_events()

        # returns the proposition that currently holds according to the grounder
        else:
            with torch.no_grad():
                img = torch.tensor(self.current_obs, device=self.sym_grounder.device).unsqueeze(0)
                pred_sym = torch.argmax(self.sym_grounder(img), dim=-1)[0]
            return self.dictionary_symbols[pred_sym]



# Preconstructed Environments

class GridWorldEnv_Base(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            randomize_loc = True,
            wrap_around_map = True,
            agent_centric_view = False
        )


class GridWorldEnv_Base_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = False
        )


class GridWorldEnv_AgentCentric(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            randomize_loc = True,
            wrap_around_map = True,
            agent_centric_view = True
        )


class GridWorldEnv_AgentCentric_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = True
        )


class GridWorldEnv_NoWrapAround(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e'],
            randomize_loc = True,
            wrap_around_map = False,
            agent_centric_view = False
        )


class GridWorldEnv_NoWrapAround_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = False,
            wrap_around_map = False,
            agent_centric_view = False
        )


class GridWorldEnv_12_Base(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            #symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            #symbols = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"],
            symbols = ['c0', 'c1', 'c2', 'c3'],
            randomize_loc = True,
            wrap_around_map = True,
            agent_centric_view = False
        )



class GridWorldEnv_12_Base_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = False
        )


class GridWorldEnv_12_AgentCentric(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = True,
            wrap_around_map = True,
            agent_centric_view = True
        )


class GridWorldEnv_12_AgentCentric_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = False,
            wrap_around_map = True,
            agent_centric_view = True
        )


class GridWorldEnv_12_NoWrapAround(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = True,
            wrap_around_map = False,
            agent_centric_view = False
        )


class GridWorldEnv_12_NoWrapAround_FixedMap(GridWorldEnv_LTL2Action):
    def __init__(self, state_type, grounder, obs_size):
        super().__init__(
            state_type = state_type,
            grounder = grounder,
            obs_size = obs_size,
            symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
            randomize_loc = False,
            wrap_around_map = False,
            agent_centric_view = False
        )