"""Native contract tests: real upstream classes, fake LLM/storage, no inference."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import pytest

CHECKOUT = os.environ.get('MEMORYBENCH_CHECKOUT')
if not CHECKOUT:
    pytest.skip('Set MEMORYBENCH_CHECKOUT to the pinned installed checkout', allow_module_level=True)
sys.path.insert(0, CHECKOUT)
from src.agent.mempp import MemppAgent, MemppAgentConfig
from src.solver.mempp import MemppSolver


class FakeStore:
    worlds = {}
    fail_after = None
    def __init__(self, world): self.world=world
    def records(self): return self.worlds.setdefault(self.world,{})
    def record_id(self, key): return str(uuid.uuid5(uuid.UUID(self.world),key))
    def put(self,key,content):
        if self.fail_after is not None and len(self.records())>=self.fail_after:
            self.fail_after=None
            raise RuntimeError('synthetic interruption')
        if key in self.records() and self.records()[key]!=content: raise ValueError('conflict')
        self.records()[key]=content
    def search(self,query,k): return [{'id':self.record_id(key),'content':'user: '+value} for key,value in sorted(self.records().items())][:k]
    def verify(self,documents,*,allow_subset=False):
        actual=self.records()
        if (not allow_subset and actual!=documents) or any(k not in documents or documents[k]!=v for k,v in actual.items()):
            raise RuntimeError('snapshot mismatch')


class LLM:
    def generate_response(self,**kwargs):self.request=kwargs;return 'SYNTHETIC CONTRACT RESPONSE'


@pytest.fixture(autouse=True)
def clear_store():FakeStore.worlds={}


@pytest.fixture
def factory(tmp_path):
    run=str(uuid.uuid4())
    def build(name='train',run_id=run):
        return MemppAgent(MemppAgentConfig(memory_cache_dir=str(tmp_path/name),run_id=run_id),store_factory=FakeStore,llm=LLM())
    return build


def test_reentrant_ingest_and_revised_observations(factory):
    a=factory();messages=[{'role':'user','content':'Question'},{'role':'assistant','content':'First response'}]
    a.add_conversation_to_memory(messages,'x');a.add_conversation_to_memory(messages,'x')
    assert len(a.export_memory_records())==2
    messages[1]['content']='Revised response';a.add_conversation_to_memory(messages,'x')
    assert len(a.export_memory_records())==3
    assert any('First response' in r['content'] for r in a.export_memory_records())


def test_save_load_roundtrip(factory):
    a=factory();a.add_memory('Original source','one');a.save_memories()
    b=factory();b.load_memories();b.load_memories()
    assert b.export_memory_records()==a.export_memory_records()


def test_corpus_forks_preserve_training_and_separate_mutations(factory):
    a=factory();a.add_memory('Shared training feedback','train');a.save_memories()
    for name in ['corpusA','corpusB']:
        b=factory(name);b.cache_path.mkdir();shutil.copyfile(a.manifest,b.manifest);b.load_memories();b.load_memories()
        assert 'Shared training feedback' in '\n'.join(b.retrieve_memory('training'))
        b.add_memory(name,'corpus')
    left=factory('corpusA');right=factory('corpusB')
    # Corpus snapshots are saved explicitly when checkpointing an evolved world.
    for b in [left,right]:
        # Existing live state intentionally differs from training-only manifest.
        with pytest.raises(RuntimeError):b.load_memories()
    assert set(a.store.records())=={'train'}
    assert FakeStore.worlds[left.world]['corpus']=='corpusA'
    assert FakeStore.worlds[right.world]['corpus']=='corpusB'


def test_interrupted_fork_can_retry_without_duplicates(factory):
    a=factory();a.add_memory('one','one');a.add_memory('two','two');a.save_memories()
    b=factory('fork');b.cache_path.mkdir();shutil.copyfile(a.manifest,b.manifest);b.store.fail_after=1
    with pytest.raises(RuntimeError):b.load_memories()
    b.load_memories();assert len(b.export_memory_records())==2


@pytest.mark.parametrize('mode',['wrong_run','missing_db','tampered_file','stale_db'])
def test_cache_failures_are_not_silently_loaded(factory,mode):
    a=factory();a.add_memory('one','one');a.save_memories()
    if mode=='wrong_run':
        b=factory(run_id=str(uuid.uuid4()))
    else:b=factory()
    if mode=='missing_db':b.store.records().clear()
    if mode=='tampered_file':
        value=json.loads(b.manifest.read_text());value['payload']['documents']['one']='changed';b.manifest.write_text(json.dumps(value))
    if mode=='stale_db':b.manifest.unlink()
    with pytest.raises((ValueError,RuntimeError)):b.retrieve_memory('one')


def test_generate_preserves_augmented_history_for_native_followups(factory):
    a=factory();a.add_conversation_to_memory([{'role':'system','content':'Allowed training feedback'}],0)
    messages=[{'role':'user','content':'Question?'}];before=copy.deepcopy(messages)
    assert a.generate_response(messages)=='SYNTHETIC CONTRACT RESPONSE'
    assert before[0]['content']=='Question?'
    assert messages[0]['content'].startswith('Context:')
    assert 'Speaker system says: Allowed training feedback' in a.llm.request['messages'][-1]['content']
    assert len(a.get_last_memory_trace())==1


def test_source_identity_conflict(factory):
    a=factory();a.add_memory('one','id')
    with pytest.raises(ValueError):a.add_memory('changed','id')


def test_solver_namespaces_pooled_training_dataset_indices(factory,monkeypatch):
    a=factory();solver=object.__new__(MemppSolver);solver.agent=a
    captured=[]
    monkeypatch.setattr(solver,'_create_or_load_memory',lambda rows,can_thread:captured.extend(rows))
    solver.create_or_load_memory([{'dataset':'A','test_idx':1,'dialog':[]},{'dataset':'B','test_idx':1,'dialog':[]}])
    assert captured[0]['test_idx']!=captured[1]['test_idx']


def test_solver_corpus_sessions_and_prediction_failure(factory):
    a=factory();solver=object.__new__(MemppSolver);solver.agent=a
    solver.memory_locomo_conversation({'session_1_date_time':'source date','session_1':[{'dia_id':'D1:1','speaker':'Alice','text':'original turn'}]},1)
    assert any('Alice says: original turn' in r['content'] for r in a.export_memory_records())
    class Dataset:
        def get_initial_chat_messages(self,idx):raise RuntimeError('synthetic infrastructure failure')
    with pytest.raises(RuntimeError):solver.predict_single_data(Dataset(),{'test_idx':3,'lang':'en'})


def test_snapshot_does_not_save_credentials(factory):
    a=factory();a.config.llm_config['api_key']='SYNTHETIC-MUST-NOT-PERSIST';a.save_memories()
    assert 'SYNTHETIC-MUST-NOT-PERSIST' not in a.manifest.read_text()


def test_native_registry_and_factory(factory,monkeypatch,tmp_path):
    from src import memory_systems
    from src.solver import SolverFactory
    real=MemppAgent.__init__
    monkeypatch.setattr(MemppAgent,'__init__',lambda self,config:real(self,config,store_factory=FakeStore,llm=LLM()))
    solver=SolverFactory.create('mempp',{'run_id':str(uuid.uuid4())},memory_cache_dir=str(tmp_path/'native'))
    assert isinstance(solver,MemppSolver)
    assert 'mempp' not in memory_systems.on_policy_names_with_memory()
    assert memory_systems.get('mempp').dialog_key()=='dialog_mempp'


def test_native_provider_omits_unset_sampling_and_internal_annotations():
    from src.llms.anthropic import AnthropicLLM
    from types import SimpleNamespace
    provider=object.__new__(AnthropicLLM)
    provider.config=SimpleNamespace(model='claude-sonnet-4-6',max_tokens=4096,temperature=1,top_p=None)
    class Messages:
        def create(self,**kwargs):self.kwargs=kwargs;return SimpleNamespace(content=[SimpleNamespace(text='SYNTHETIC')])
    api=Messages();provider.client=SimpleNamespace(messages=api)
    provider.generate_response([{'role':'user','content':'test','origin_content':'internal metadata'}])
    assert 'top_p' not in api.kwargs
    assert set(api.kwargs['messages'][0])=={'role','content'}


def test_native_reading_retry_keeps_original_question(factory):
    a=factory();a.add_memory('source','source')
    queries=[]
    search=a.store.search
    def capture(query,k):
        queries.append(query)
        return search(query,k)
    a.store.search=capture
    class OnceFailingLLM:
        calls=0
        def generate_response(self,**kwargs):
            self.calls+=1
            if self.calls==1:raise RuntimeError('synthetic timeout')
            return 'SYNTHETIC RETRY'
    a.llm=OnceFailingLLM()
    messages=[{'role':'user','content':'Original question','origin_content':'Original question'}]
    with pytest.raises(RuntimeError):a.generate_response(messages)
    assert messages[-1]['content']=='Original question'
    assert a.generate_response(messages)=='SYNTHETIC RETRY'
    assert queries==['Original question','Original question']
    assert messages[-1]['content'].count('Context:')==1


def test_native_training_keeps_observed_context_and_original_roles(factory):
    a=factory()
    a.add_conversation_to_memory([{'role':'user','content':'Context: Allowed observed training context User:Q','origin_content':'Q'}],'A:1')
    records=a.export_memory_records()
    assert len(records)==1
    assert records[0]['content']=='Speaker user says: Context: Allowed observed training context User:Q'
