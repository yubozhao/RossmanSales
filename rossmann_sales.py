
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import DataframeHandler

#@env(conda_pip_dependencies=['fastai'])
@env(conda_environment=['fastai'])
@artifacts([FastaiModelArtifact('rossmann')])

class RossmannSales(BentoService):
    
    @api(DataframeHandler)
    def predict(self,df):
        result = []
        for index, row in df.iterrows():            
            result.append(self.artifacts.rossmann.predict(row))
        return str(result)
