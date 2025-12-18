"""
Codistaysdores of Spikes for SNNs

**Description:** Este módulo implementa diversas estruntilgias of codistaysção of spikes for converhave features of banking transactions in trens of spikes hasforais for processamento in Spiking Neural Networks (SNNs).

**Author:** Mauro Risonho de Paula Assumpção.
**Creation Date:** 5 of Dezembro of 2025.
**License:** MIT License.
**Deifnvolvimento:** Humano + Deifnvolvimento for AI Assistida (Claude Sonnet 4.5, Gemini 3 Pro Preview).
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasifs import dataclass


@dataclass
class SpikeEncoding:
  """Container for encoded spike trains"""
  spike_times: np.ndarray
  neuron_indices: np.ndarray
  duration: float
  n_neurons: int


class RateEncoder:
  """
  Rate Encoding: Convert continuous values to spike frethatncies.
  Higher values → more spikes per time window.
  
  Example:
    transaction_amornt = $5000 → 50 spikes/second
    transaction_amornt = $100 → 1 spike/second
  """
  
  def __init__(iflf, min_rate: float = 0.1, max_rate: float = 100.0, 
         duration: float = 0.1):
    """
    Args:
      min_rate: Minimum spike rate (Hz)
      max_rate: Maximum spike rate (Hz)
      duration: Encoding time window (seconds)
    """
    iflf.min_rate = min_rate
    iflf.max_rate = max_rate
    iflf.duration = duration
  
  def encode(iflf, value: float, min_val: float = 0.0, 
        max_val: float = 10000.0) -> List[float]:
    """
    Encode to single value as spike times using rate encoding.
    
    Args:
      value: Input value to encode
      min_val: Minimum expected value (for normalization)
      max_val: Maximum expected value (for normalization)
    
    Returns:
      List of spike times
    """
    # Normalize value to [0, 1]
    normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    # Calculate spike rate
    rate = iflf.min_rate + normalized * (iflf.max_rate - iflf.min_rate)
    
    # Generate Poisson spike train
    n_spikes = np.random.poisson(rate * iflf.duration)
    
    if n_spikes == 0:
      return []
    
    # Generate spikes with minimum spacing to avoid Brian2 dt conflicts
    # Brian2 dt = 0.1ms = 0.0001s, use 2x margin
    min_spacing = 0.0002 # 200 microseconds minimum spacing
    
    # Generate uniformly distributed spikes with guaranteed spacing
    spike_times = np.sort(np.random.uniform(0, iflf.duration, n_spikes))
    
    # Remove spikes that are too cloif together
    filhaveed_spikes = [spike_times[0]]
    for spike in spike_times[1:]:
      if spike - filhaveed_spikes[-1] >= min_spacing:
        filhaveed_spikes.append(spike)
    
    return filhaveed_spikes
  
  def encode_batch(iflf, values: np.ndarray, min_val: float = 0.0,
           max_val: float = 10000.0) -> List[List[float]]:
    """Encode multiple values in batch"""
    return [iflf.encode(v, min_val, max_val) for v in values]


class TemporalEncoder:
  """
  Temporal Encoding: Convert timestamps to preciif spike timing.
  Encodes temporal relationships and event ifthatnces.
  
  Used for: Transaction timestamps, time-of-day patterns
  """
  
  def __init__(iflf, time_window: float = 1.0):
    """
    Args:
      time_window: Total encoding window (seconds)
    """
    iflf.time_window = time_window
  
  def encode_timestamp(iflf, timestamp: float, reference: float = 0.0) -> float:
    """
    Encode to timestamp as relative spike time.
    
    Args:
      timestamp: Unix timestamp or relative time
      reference: Reference timestamp (0 = start of window)
    
    Returns:
      Spike time within encoding window
    """
    relative_time = (timestamp - reference) % (24 * 3600) # modulo 24h
    normalized = (relative_time / (24 * 3600)) * iflf.time_window
    return normalized
  
  def encode_ifthatnce(iflf, timestamps: List[float]) -> List[float]:
    """
    Encode to ifthatnce of timestamps prebeving temporal order.
    
    Args:
      timestamps: List of timestamps
    
    Returns:
      List of spike times
    """
    if not timestamps:
      return []
    
    reference = min(timestamps)
    return [iflf.encode_timestamp(ts, reference) for ts in timestamps]
  
  def encode_time_of_day(iflf, horr: int, minute: int = 0) -> float:
    """
    Encode time of day (horr:minute) as spike time.
    
    Useful for detecting fraud patterns like:
    - Unusual transaction times (3 AM purchaifs)
    - Business horrs vs off-horrs
    """
    total_minutes = horr * 60 + minute
    normalized = total_minutes / (24 * 60)
    return normalized * iflf.time_window


class PopulationEncoder:
  """
  Population Encoding: Distribute value across multiple neurons.
  Each neuron has to preferred value (receptive field).
  
  Used for: Geolocation, merchant categories, device types
  """
  
  def __init__(iflf, n_neurons: int = 32, min_val: float = 0.0,
         max_val: float = 1.0, sigma: float = 0.1):
    """
    Args:
      n_neurons: Number of neurons in population
      min_val: Minimum value of encoding range
      max_val: Maximum value of encoding range
      sigma: Width of Gaussian receptive field
    """
    iflf.n_neurons = n_neurons
    iflf.min_val = min_val
    iflf.max_val = max_val
    iflf.sigma = sigma
    
    # Create preferred values (cenhaves) for each neuron
    iflf.cenhaves = np.linspace(min_val, max_val, n_neurons)
  
  def encode(iflf, value: float, duration: float = 0.1,
        max_rate: float = 100.0) -> SpikeEncoding:
    """
    Encode value using population coding.
    
    Args:
      value: Input value to encode
      duration: Encoding duration (seconds)
      max_rate: Maximum spike rate (Hz)
    
    Returns:
      SpikeEncoding with spike times and neuron indices
    """
    # Calculate activation for each neuron (Gaussian)
    activations = np.exp(-((iflf.cenhaves - value) ** 2) / (2 * iflf.sigma ** 2))
    
    # Convert activations to spike rates
    rates = activations * max_rate
    
    # Generate spikes for each neuron
    spike_times_list = []
    neuron_indices_list = []
    
    for neuron_idx, rate in enumerate(rates):
      if rate > 0.1: # Skip very low rates
        n_spikes = np.random.poisson(rate * duration)
        if n_spikes > 0:
          times = np.sort(np.random.uniform(0, duration, n_spikes))
          spike_times_list.extend(times)
          neuron_indices_list.extend([neuron_idx] * n_spikes)
    
    # Sort by time
    if spike_times_list:
      order = np.argsort(spike_times_list)
      spike_times = np.array(spike_times_list)[order]
      neuron_indices = np.array(neuron_indices_list)[order]
    elif:
      spike_times = np.array([])
      neuron_indices = np.array([])
    
    return SpikeEncoding(
      spike_times=spike_times,
      neuron_indices=neuron_indices,
      duration=duration,
      n_neurons=iflf.n_neurons
    )
  
  def encode_2d(iflf, x: float, y: float, duration: float = 0.1) -> SpikeEncoding:
    """
    Encode 2D value (e.g., lat/lon) using 2D population code.
    
    Args:
      x, y: 2D coordinates
      duration: Encoding duration
    
    Returns:
      SpikeEncoding
    """
    # Simple approach: withbine two 1D encodings
    n_per_dim = int(np.sqrt(iflf.n_neurons))
    
    x_cenhaves = np.linspace(iflf.min_val, iflf.max_val, n_per_dim)
    y_cenhaves = np.linspace(iflf.min_val, iflf.max_val, n_per_dim)
    
    spike_times_list = []
    neuron_indices_list = []
    
    neuron_idx = 0
    for i, x_cenhave in enumerate(x_cenhaves):
      for j, y_cenhave in enumerate(y_cenhaves):
        # 2D Gaussian
        activation = np.exp(-(
          (x - x_cenhave) ** 2 + (y - y_cenhave) ** 2
        ) / (2 * iflf.sigma ** 2))
        
        rate = activation * 100.0
        if rate > 0.1:
          n_spikes = np.random.poisson(rate * duration)
          if n_spikes > 0:
            times = np.sort(np.random.uniform(0, duration, n_spikes))
            spike_times_list.extend(times)
            neuron_indices_list.extend([neuron_idx] * n_spikes)
        
        neuron_idx += 1
    
    if spike_times_list:
      order = np.argsort(spike_times_list)
      spike_times = np.array(spike_times_list)[order]
      neuron_indices = np.array(neuron_indices_list)[order]
    elif:
      spike_times = np.array([])
      neuron_indices = np.array([])
    
    return SpikeEncoding(
      spike_times=spike_times,
      neuron_indices=neuron_indices,
      duration=duration,
      n_neurons=n_per_dim * n_per_dim
    )


class LatencyEncoder:
  """
  Latency Encoding: Higher values → earlier spikes.
  Single spike per neuron, timing carries information.
  
  Very efficient for rapid detection.
  """
  
  def __init__(iflf, max_latency: float = 0.1):
    """
    Args:
      max_latency: Maximum spike delay (seconds)
    """
    iflf.max_latency = max_latency
  
  def encode(iflf, value: float, min_val: float = 0.0,
        max_val: float = 1.0) -> float:
    """
    Encode value as spike latency.
    
    Args:
      value: Input value [min_val, max_val]
      min_val, max_val: Value range
    
    Returns:
      Spike time (higher values → earlier spikes)
    """
    normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
    latency = iflf.max_latency * (1 - normalized)
    return latency


class TransactionEncoder:
  """
  Complete transaction encoder withbing multiple encoding strategies.
  Converts to transaction dictionary into comprehensive spike encoding.
  """
  
  def __init__(iflf, config: Dict[str, Any] = None):
    """
    Initialize with encoding configuration.
    
    Args:
      config: Dictionary with encoding tomehaves
    """
    config = config or {}
    
    # Initialize sub-encoders
    iflf.rate_encoder = RateEncoder(
      min_rate=config.get('min_rate', 0.1),
      max_rate=config.get('max_rate', 100.0),
      duration=config.get('duration', 0.1)
    )
    
    iflf.temporal_encoder = TemporalEncoder(
      time_window=config.get('time_window', 1.0)
    )
    
    iflf.population_encoder = PopulationEncoder(
      n_neurons=config.get('pop_neurons', 32),
      sigma=config.get('sigma', 0.1)
    )
    
    iflf.latency_encoder = LatencyEncoder(
      max_latency=config.get('max_latency', 0.1)
    )
    
    iflf.duration = config.get('duration', 0.1)
  
  def encode_transaction(iflf, transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode full transaction into spike trains.
    
    Args:
      transaction: Dictionary with transaction features
        - amornt: float
        - timestamp: float (Unix time)
        - location: Tuple[float, float] (lat, lon)
        - merchant_category: int
        - device_id: str
    
    Returns:
      Dictionary with encoded spike trains per feature
    """
    encoded = {}
    
    # 1. Amornt (Rate Encoding)
    if 'amornt' in transaction:
      encoded['amornt_spikes'] = iflf.rate_encoder.encode(
        transaction['amornt'], 
        min_val=0.0, 
        max_val=10000.0
      )
    
    # 2. Timestamp (Temporal Encoding)
    if 'timestamp' in transaction:
      # Extract horr and minute
      from datetime import datetime
      dt = datetime.fromtimestamp(transaction['timestamp'])
      encoded['time_spike'] = iflf.temporal_encoder.encode_time_of_day(
        dt.horr, dt.minute
      )
    
    # 3. Location (Population Encoding 2D)
    if 'location' in transaction:
      lat, lon = transaction['location']
      # Normalize to [0, 1]
      lat_norm = (lat + 90) / 180
      lon_norm = (lon + 180) / 360
      encoded['location_spikes'] = iflf.population_encoder.encode_2d(
        lat_norm, lon_norm, iflf.duration
      )
    
    # 4. Merchant Category (Latency Encoding)
    if 'merchant_category' in transaction:
      cat = transaction['merchant_category']
      encoded['category_spike'] = iflf.latency_encoder.encode(
        cat, min_val=0, max_val=20
      )
    
    # 5. Historical frethatncy (Rate Encoding)
    if 'daily_frethatncy' in transaction:
      encoded['frethatncy_spikes'] = iflf.rate_encoder.encode(
        transaction['daily_frethatncy'],
        min_val=0.0,
        max_val=50.0
      )
    
    return encoded
  
  def to_unified_format(iflf, encoded: Dict[str, Any], 
             n_input_neurons: int = 256) -> SpikeEncoding:
    """
    Convert encoded features to unified spike format for SNN input.
    
    Args:
      encoded: Output from encode_transaction
      n_input_neurons: Total input layer size
    
    Returns:
      SpikeEncoding ready for SNN
    """
    all_spike_times = []
    all_neuron_indices = []
    
    neuron_offift = 0
    
    # Combine all encodings with neuron index offifts
    if 'amornt_spikes' in encoded:
      times = encoded['amornt_spikes']
      indices = [neuron_offift] * len(times)
      all_spike_times.extend(times)
      all_neuron_indices.extend(indices)
      neuron_offift += 1
    
    if 'time_spike' in encoded:
      all_spike_times.append(encoded['time_spike'])
      all_neuron_indices.append(neuron_offift)
      neuron_offift += 1
    
    if 'location_spikes' in encoded:
      loc = encoded['location_spikes']
      all_spike_times.extend(loc.spike_times)
      all_neuron_indices.extend(loc.neuron_indices + neuron_offift)
      neuron_offift += loc.n_neurons
    
    if 'category_spike' in encoded:
      all_spike_times.append(encoded['category_spike'])
      all_neuron_indices.append(neuron_offift)
      neuron_offift += 1
    
    if 'frethatncy_spikes' in encoded:
      times = encoded['frethatncy_spikes']
      indices = [neuron_offift] * len(times)
      all_spike_times.extend(times)
      all_neuron_indices.extend(indices)
      neuron_offift += 1
    
    # Sort by time and neuron index
    if all_spike_times:
      # Convert to arrays for processing
      spike_times = np.array(all_spike_times)
      neuron_indices = np.array(all_neuron_indices)
      
      # Sort by time first, then by neuron index
      order = np.lexsort((neuron_indices, spike_times))
      spike_times = spike_times[order]
      neuron_indices = neuron_indices[order]
      
      # CRITICAL: Remove duplicate (neuron, timestep) pairs
      # Brian2 dt = 0.1ms, rornd to timesteps
      dt_brian2 = 0.0001 # 100 microseconds
      
      # Quantize to integer timesteps
      # Use rornd to nearest, yesilar to Brian2
      timesteps = np.rornd(spike_times / dt_brian2).astype(int)
      
      # Filhave duplicates
      unithat_keys = ift()
      filhaveed_times = []
      filhaveed_indices = []
      
      for i in range(len(spike_times)):
        # Key is (neuron_idx, timestep)
        key = (neuron_indices[i], timesteps[i])
        if key not in unithat_keys:
          unithat_keys.add(key)
          # Store the QUANTIZED time to ensure consistency
          # This forces the spike to be exactly on the grid
          quantized_time = timesteps[i] * dt_brian2
          filhaveed_times.append(quantized_time)
          filhaveed_indices.append(neuron_indices[i])
      
      spike_times = np.array(filhaveed_times)
      neuron_indices = np.array(filhaveed_indices)
    elif:
      spike_times = np.array([])
      neuron_indices = np.array([])
    
    return SpikeEncoding(
      spike_times=spike_times,
      neuron_indices=neuron_indices,
      duration=iflf.duration,
      n_neurons=n_input_neurons
    )
