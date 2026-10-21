import os, csv, io, uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Header, Response, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, String, Integer, Numeric, ForeignKey, DateTime, Boolean, select, func, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
from passlib.context import CryptContext
from jose import jwt, JWTError

DB=os.getenv('DATABASE_URL','sqlite:///./takia.db')
# Railway supplies postgresql://; SQLAlchemy requires an explicit psycopg dialect.
if DB.startswith('postgres://'):
 DB='postgresql+psycopg://'+DB[len('postgres://'):]
elif DB.startswith('postgresql://'):
 DB='postgresql+psycopg://'+DB[len('postgresql://'):]
engine=create_engine(DB,connect_args={'check_same_thread':False} if DB.startswith('sqlite') else {},pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine)
SECRET=os.getenv('SECRET_KEY','DEV_ONLY_NOT_FOR_PRODUCTION')
if SECRET=='DEV_ONLY_NOT_FOR_PRODUCTION' and not DB.startswith('sqlite'): raise RuntimeError('Set SECRET_KEY')
pwd=CryptContext(schemes=['bcrypt'],deprecated='auto')
class Base(DeclarativeBase): pass
class User(Base):
 __tablename__='users'
 id:Mapped[int]=mapped_column(primary_key=True)
 email:Mapped[str]=mapped_column(String(180),unique=True,index=True)
 password_hash:Mapped[str]=mapped_column(String(255))
 role:Mapped[str]=mapped_column(String(20),default='cashier')
 active:Mapped[bool]=mapped_column(Boolean,default=True)
class Outlet(Base):
 __tablename__='outlets'
 id:Mapped[int]=mapped_column(primary_key=True)
 name:Mapped[str]=mapped_column(String(100),unique=True)
class Product(Base):
 __tablename__='products'
 id:Mapped[int]=mapped_column(primary_key=True)
 name:Mapped[str]=mapped_column(String(160))
 category:Mapped[str]=mapped_column(String(100),default='General')
 price:Mapped[Decimal]=mapped_column(Numeric(12,2))
 stock:Mapped[Decimal]=mapped_column(Numeric(12,3),default=0)
 active:Mapped[bool]=mapped_column(Boolean,default=True)
class Sale(Base):
 __tablename__='sales'
 id:Mapped[int]=mapped_column(primary_key=True)
 public_id:Mapped[str]=mapped_column(String(36),unique=True,default=lambda:str(uuid.uuid4()))
 idempotency_key:Mapped[str]=mapped_column(String(100),unique=True)
 outlet_id:Mapped[int]=mapped_column(ForeignKey('outlets.id'))
 cashier_id:Mapped[int]=mapped_column(ForeignKey('users.id'))
 method:Mapped[str]=mapped_column(String(20))
 subtotal:Mapped[Decimal]=mapped_column(Numeric(12,2))
 tax:Mapped[Decimal]=mapped_column(Numeric(12,2))
 total:Mapped[Decimal]=mapped_column(Numeric(12,2))
 room_number:Mapped[Optional[str]]=mapped_column(String(30),nullable=True)
 guest_name:Mapped[Optional[str]]=mapped_column(String(160),nullable=True)
 integration_status:Mapped[str]=mapped_column(String(30),default='not_applicable')
 fiscal_status:Mapped[str]=mapped_column(String(30),default='not_fiscalised')
 created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
 items:Mapped[list['SaleItem']]=relationship(cascade='all,delete-orphan')
class SaleItem(Base):
 __tablename__='sale_items'
 id:Mapped[int]=mapped_column(primary_key=True)
 sale_id:Mapped[int]=mapped_column(ForeignKey('sales.id'))
 product_id:Mapped[int]=mapped_column(ForeignKey('products.id'))
 name:Mapped[str]=mapped_column(String(160))
 quantity:Mapped[Decimal]=mapped_column(Numeric(12,3))
 unit_price:Mapped[Decimal]=mapped_column(Numeric(12,2))
 line_total:Mapped[Decimal]=mapped_column(Numeric(12,2))
class StockEvent(Base):
 __tablename__='stock_events'
 id:Mapped[int]=mapped_column(primary_key=True)
 product_id:Mapped[int]=mapped_column(ForeignKey('products.id'))
 change:Mapped[Decimal]=mapped_column(Numeric(12,3))
 reason:Mapped[str]=mapped_column(String(120))
 actor_id:Mapped[int]=mapped_column(ForeignKey('users.id'))
 created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
class Audit(Base):
 __tablename__='audit'
 id:Mapped[int]=mapped_column(primary_key=True)
 actor_id:Mapped[int]=mapped_column(ForeignKey('users.id'))
 action:Mapped[str]=mapped_column(String(100))
 detail:Mapped[str]=mapped_column(String(500))
 created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))

app=FastAPI(title='Hotel Takia CloudPOS',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:8000').split(','),allow_credentials=False,allow_methods=['*'],allow_headers=['*'])
@app.on_event('startup')
def startup():
 Base.metadata.create_all(engine)
 with SessionLocal() as db:
  for name in ['Bar','Restaurant','Reception']:
   if not db.scalar(select(Outlet).where(Outlet.name==name)):db.add(Outlet(name=name))
  email=os.getenv('ADMIN_EMAIL'); password=os.getenv('ADMIN_PASSWORD')
  if email and password and not db.scalar(select(User).where(User.email==email)):
   if len(password)<12:raise RuntimeError('ADMIN_PASSWORD must be at least 12 characters')
   db.add(User(email=email,password_hash=pwd.hash(password),role='admin'))
  db.commit()
def session():
 with SessionLocal() as db: yield db

def current(authorization:Optional[str]=Header(None),db:Session=Depends(session)):
 if not authorization or not authorization.startswith('Bearer '):raise HTTPException(401,'Login required')
 try: sub=jwt.decode(authorization[7:],SECRET,algorithms=['HS256'])['sub']; u=db.get(User,int(sub))
 except (JWTError,KeyError,ValueError):raise HTTPException(401,'Invalid token')
 if not u or not u.active:raise HTTPException(401,'Invalid user')
 return u

def manager(u:User=Depends(current)):
 if u.role not in ('admin','manager'):raise HTTPException(403,'Manager access required')
 return u

def money(v):return Decimal(str(v)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
def sale_dict(s):
 return dict(id=s.id,reference=s.public_id,outlet_id=s.outlet_id,method=s.method,subtotal=str(s.subtotal),tax=str(s.tax),total=str(s.total),room_number=s.room_number,guest_name=s.guest_name,integration_status=s.integration_status,fiscal_status=s.fiscal_status,created_at=s.created_at.isoformat(),items=[dict(name=i.name,quantity=str(i.quantity),unit_price=str(i.unit_price),line_total=str(i.line_total)) for i in s.items])
class Login(BaseModel): email:str; password:str
class ProductIn(BaseModel):name:str=Field(min_length=1,max_length=160);category:str='General';price:Decimal=Field(ge=0);stock:Decimal=Field(default=0,ge=0)
class ItemIn(BaseModel):product_id:int;quantity:Decimal=Field(gt=0,le=10000)
class SaleIn(BaseModel):idempotency_key:str=Field(min_length=8,max_length=100);outlet_id:int;method:str;items:list[ItemIn]=Field(min_length=1);room_number:Optional[str]=None;guest_name:Optional[str]=None
class StockIn(BaseModel):change:Decimal;reason:str=Field(min_length=3,max_length=120)
class UserIn(BaseModel):email:str;password:str=Field(min_length=12);role:str='cashier'
@app.get('/')
def home():return FileResponse(Path(__file__).parent/'static/index.html')
@app.get('/static/app.js')
def js():return FileResponse(Path(__file__).parent/'static/app.js',media_type='application/javascript')
@app.get('/health')
def health():return {'status':'ok'}
@app.post('/api/login')
def login(body:Login,db:Session=Depends(session)):
 u=db.scalar(select(User).where(User.email==body.email))
 if not u or not pwd.verify(body.password,u.password_hash) or not u.active:raise HTTPException(401,'Invalid credentials')
 token=jwt.encode({'sub':str(u.id),'exp':datetime.now(timezone.utc)+timedelta(hours=8)},SECRET,algorithm='HS256')
 return {'token':token,'role':u.role,'email':u.email}
@app.get('/api/outlets')
def outlets(db:Session=Depends(session),u:User=Depends(current)):
 return [{'id':o.id,'name':o.name} for o in db.scalars(select(Outlet).order_by(Outlet.id))]
@app.get('/api/products')
def products(db:Session=Depends(session),u:User=Depends(current)):
 return [{'id':p.id,'name':p.name,'category':p.category,'price':str(p.price),'stock':str(p.stock)} for p in db.scalars(select(Product).where(Product.active==True).order_by(Product.name))]
@app.post('/api/products',status_code=201)
def add_product(body:ProductIn,db:Session=Depends(session),u:User=Depends(manager)):
 p=Product(**body.model_dump());db.add(p);db.flush();db.add(Audit(actor_id=u.id,action='product_create',detail=f'{p.id}: {p.name}'));db.commit();return {'id':p.id}
@app.post('/api/products/{product_id}/stock')
def stock(product_id:int,body:StockIn,db:Session=Depends(session),u:User=Depends(manager)):
 p=db.get(Product,product_id)
 if not p:raise HTTPException(404,'Product not found')
 if p.stock+body.change<0:raise HTTPException(409,'Insufficient stock')
 p.stock+=body.change;db.add(StockEvent(product_id=p.id,change=body.change,reason=body.reason,actor_id=u.id));db.add(Audit(actor_id=u.id,action='stock_adjustment',detail=f'{p.id}: {body.change} {body.reason}'));db.commit();return {'stock':str(p.stock)}
@app.post('/api/users',status_code=201)
def create_user(body:UserIn,db:Session=Depends(session),u:User=Depends(manager)):
 if body.role not in ('cashier','manager') or (body.role=='manager' and u.role!='admin'):raise HTTPException(403,'Role not permitted')
 if db.scalar(select(User).where(User.email==body.email)):raise HTTPException(409,'Email already exists')
 x=User(email=body.email,password_hash=pwd.hash(body.password),role=body.role);db.add(x);db.commit();return {'id':x.id}
@app.post('/api/sales',status_code=201)
def create_sale(body:SaleIn,db:Session=Depends(session),u:User=Depends(current)):
 existing=db.scalar(select(Sale).where(Sale.idempotency_key==body.idempotency_key))
 if existing:return sale_dict(existing)
 if body.method not in ('cash','eftpos','card','room'):raise HTTPException(422,'Unsupported payment method')
 if body.method=='room' and (not body.room_number or not body.guest_name):raise HTTPException(422,'Room and guest are required')
 if not db.get(Outlet,body.outlet_id):raise HTTPException(404,'Outlet not found')
 quantities={}
 for item in body.items:quantities[item.product_id]=quantities.get(item.product_id,Decimal('0'))+item.quantity
 products={pid:db.get(Product,pid) for pid in quantities}
 for pid,qty in quantities.items():
  p=products[pid]
  if not p or not p.active:raise HTTPException(404,f'Product {pid} not found')
  if p.stock<qty:raise HTTPException(409,f'Insufficient stock: {p.name}')
 # Prices are VAT-inclusive; configurable tax rules and actual FRCS fiscalisation remain future work.
 tax_rate=Decimal('0.125') # Demonstration default only; confirm current Fiji tax rates before live use.
 total=sum((money(products[pid].price*qty) for pid,qty in quantities.items()),Decimal('0'))
 tax=money(total*tax_rate/(Decimal('1')+tax_rate));subtotal=total-tax
 s=Sale(idempotency_key=body.idempotency_key,outlet_id=body.outlet_id,cashier_id=u.id,method=body.method,subtotal=subtotal,tax=tax,total=total,room_number=body.room_number if body.method=='room' else None,guest_name=body.guest_name if body.method=='room' else None,integration_status='pending_preno' if body.method=='room' else 'not_applicable')
 db.add(s);db.flush()
 for pid,qty in quantities.items():
  p=products[pid];line=money(p.price*qty)
  s.items.append(SaleItem(product_id=pid,name=p.name,quantity=qty,unit_price=p.price,line_total=line))
  p.stock-=qty;db.add(StockEvent(product_id=pid,change=-qty,reason=f'Sale {s.public_id}',actor_id=u.id))
 db.add(Audit(actor_id=u.id,action='sale_created',detail=f'{s.public_id}: {s.method} {s.total}'))
 db.commit();db.refresh(s);return sale_dict(s)
@app.get('/api/sales')
def sales(limit:int=Query(50,ge=1,le=500),db:Session=Depends(session),u:User=Depends(current)):
 return [sale_dict(s) for s in db.scalars(select(Sale).order_by(Sale.id.desc()).limit(limit))]
@app.get('/api/dashboard')
def dashboard(db:Session=Depends(session),u:User=Depends(current)):
 today=datetime.now(timezone.utc).date();ss=list(db.scalars(select(Sale)))
 today_sales=[s for s in ss if s.created_at.date()==today]
 return {'sales_today':str(sum((s.total for s in today_sales),Decimal('0'))),'transactions_today':len(today_sales),'pending_room_charges':sum(s.integration_status=='pending_preno' for s in ss),'not_fiscalised':sum(s.fiscal_status=='not_fiscalised' for s in ss),'by_outlet':[{ 'outlet':o.name,'total':str(sum((s.total for s in today_sales if s.outlet_id==o.id),Decimal('0')))} for o in db.scalars(select(Outlet))]}
@app.get('/api/export/myob-daily.csv')
def myob_export(db:Session=Depends(session),u:User=Depends(manager)):
 # RECONCILIATION EXPORT ONLY, not MYOB Business's proprietary import schema or an API posting.
 rows=list(db.scalars(select(Sale).order_by(Sale.id)))
 out=io.StringIO();wr=csv.writer(out);wr.writerow(['Date UTC','Sale Ref','Outlet ID','Payment Method','Net Sales FJD','VAT FJD','Gross FJD','Room','Preno Status','Fiscal Status'])
 for s in rows:wr.writerow([s.created_at.isoformat(),s.public_id,s.outlet_id,s.method,s.subtotal,s.tax,s.total,s.room_number or '',s.integration_status,s.fiscal_status])
 out.seek(0);return StreamingResponse(iter([out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="takia-sales-reconciliation.csv"'})
