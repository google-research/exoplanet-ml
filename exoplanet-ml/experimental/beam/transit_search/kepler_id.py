# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KeplerId class for use as Beam pipeline keys."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import apache_beam as beam


class KeplerId(object):
  """Class representing a Kepler ID."""

  def __init__(self, value):
    self._value = int(value)

  @property
  def value(self):
    return self._value

  def encode(self):
    return str(self.value)

  @staticmethod
  def decode(encoded):
    return KeplerId(encoded)

  def __repr__(self):
    return "KeplerId(%s)" % self

  def __str__(self):
    return str(self.value)


class KeplerIdCoder(beam.coders.Coder):
  """A coder for reading and writing Kepler IDs."""

  def encode(self, kepler_id):
    encoded = kepler_id.encode()
    return encoded

  def decode(self, encoded):
    decoded = KeplerId.decode(encoded)
    return decoded

  def is_deterministic(self):
    return True


beam.coders.registry.register_coder(KeplerId, KeplerIdCoder)
