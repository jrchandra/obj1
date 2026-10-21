import os, tempfile
os.environ['DATABASE_URL']='sqlite:///'+tempfile.mktemp(suffix='.db')
os.environ['SECRET_KEY']='test-secret'
os.environ['ADMIN_EMAIL']='admin@example.com'
os.environ['ADMIN_PASSWORD']='testpassword123456'
from fastapi.testclient import TestClient
from app.main import app

def test_sales_and_room_charge():
 with TestClient(app) as client:
  assert client.get('/health').json()['status']=='ok'
  token=client.post('/api/login',json={'email':'admin@example.com','password':'testpassword123456'}).json()['token']
  h={'Authorization':'Bearer '+token}
  p=client.post('/api/products',headers=h,json={'name':'Test meal','price':'22.50','stock':'10','category':'Food'})
  assert p.status_code==201
  pid=p.json()['id'];outlet=client.get('/api/outlets',headers=h).json()[0]['id']
  sale={'idempotency_key':'test-sale-0001','outlet_id':outlet,'method':'room','room_number':'201','guest_name':'Test Guest','items':[{'product_id':pid,'quantity':'2'}]}
  r=client.post('/api/sales',headers=h,json=sale)
  assert r.status_code==201,r.text
  assert r.json()['total']=='45.00'
  assert r.json()['integration_status']=='pending_preno'
  assert r.json()['fiscal_status']=='not_fiscalised'
  assert client.post('/api/sales',headers=h,json=sale).json()['reference']==r.json()['reference']
  assert client.get('/api/products',headers=h).json()[0]['stock']=='8.000'
