import os
import os.path as osp
import sys
import datetime
import shutil
try:
    from mmcv import Config as MMConfig
except ImportError:
    from mmengine import Config as MMConfig

class Config:
    # Fixed by SH Heo (251230) - Pre-initialize config for multiprocessing worker compatibility
    def __init__(self):
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

        # Auto-load config from environment variable (for multiprocessing workers)
        config_path = os.environ.get('SMPLER_X_CONFIG_PATH')
        if config_path is not None and osp.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        """Load config from file without updating directories."""
        cfg = MMConfig.fromfile(config_path)
        self.__dict__.update(dict(cfg))
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        if not hasattr(self, 'data_dir') or self.data_dir is None:
            self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

    def get_config_fromfile(self, config_path):
        # Store config path in environment variable for worker processes
        abs_config_path = osp.abspath(config_path)
        os.environ['SMPLER_X_CONFIG_PATH'] = abs_config_path
        self.config_path = config_path
        cfg = MMConfig.fromfile(self.config_path)
        self.__dict__.update(dict(cfg))

        # update dir
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        # Fixed by SH Heo (251230) - Allow data_dir override from config file
        if not hasattr(self, 'data_dir') or self.data_dir is None:
            self.data_dir = osp.join(self.root_dir, 'dataset')
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')

        ## add some paths to the system root dir
        sys.path.insert(0, osp.join(self.root_dir, 'common'))
        from utils.dir import add_pypath
        add_pypath(osp.join(self.data_dir))
        for dataset in os.listdir(osp.join(self.root_dir, 'data')):
            if dataset not in ['humandata.py', '__pycache__', 'dataset.py']:
                add_pypath(osp.join(self.root_dir, 'data', dataset))
        add_pypath(osp.join(self.root_dir, 'data'))
        add_pypath(self.data_dir)
                
    def prepare_dirs(self, exp_name):
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = osp.join(self.root_dir, f'{exp_name}_{time_str}')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.code_dir = osp.join(self.output_dir, 'code')
        self.result_dir = osp.join(self.output_dir, 'result')

        from utils.dir import make_folder
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.code_dir)
        make_folder(self.result_dir)

        ## copy some code to log dir as a backup
        copy_files = ['main/train.py', 'main/test.py', 'common/base.py',
                      'common/nets', 'main/SMPLer_X.py',
                      'data/dataset.py', 'data/MSCOCO/MSCOCO.py', 'data/AGORA/AGORA.py']
        for file in copy_files:
            src = osp.join(self.root_dir, file)
            dst = osp.join(self.code_dir, osp.basename(file))
            if osp.exists(src):
                if osp.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

    def update_test_config(self, testset, agora_benchmark, shapy_eval_split, pretrained_model_path, use_cache,
                           eval_on_train=False, vis=False):
        self.testset = testset
        self.agora_benchmark = agora_benchmark
        self.pretrained_model_path = pretrained_model_path
        self.shapy_eval_split = shapy_eval_split
        self.use_cache = use_cache
        self.eval_on_train = eval_on_train
        self.vis = vis

    def update_config(self, num_gpus, exp_name):
        self.num_gpus = num_gpus
        self.exp_name = exp_name

        self.prepare_dirs(self.exp_name)

        # Save (convert backslashes to forward slashes for Windows compatibility)
        cfg_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                cfg_dict[k] = v.replace('\\', '/')
            else:
                cfg_dict[k] = v
        cfg_save = MMConfig(cfg_dict)
        cfg_save.dump(osp.join(self.code_dir,'config_base.py'))

cfg = Config()