import subprocess
subprocess.Popen(['bash','-lc',
                  'nohup python /content/perturb/_pair_driver.py > /content/perturb/pair_driver.log 2>&1 &'],
                 start_new_session=True)
print('pair driver launched')
