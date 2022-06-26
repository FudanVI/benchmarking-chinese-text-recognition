from __future__ import absolute_import

__factory = {
}

def names():
  return sorted(__factory.keys())

def factory():
  return __factory